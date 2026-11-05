// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Tests of [`ArrowReaderBuilder::with_dictionary_field_ids`](super::ArrowReaderBuilder::with_dictionary_field_ids).

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use tempfile::TempDir;

use crate::Runtime;
use crate::arrow::ArrowReaderBuilder;
use crate::expr::{Bind, Predicate, Reference};
use crate::io::FileIO;
use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
use crate::spec::{
    DataContentType, DataFileFormat, Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type,
};

const ROWS: i32 = 30;
const GROUP_ROWS: usize = 10;

fn dict_type() -> DataType {
    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
}

fn value(i: i32) -> Option<String> {
    if i % 7 == 6 {
        None
    } else {
        Some(["apple", "apricot", "banana", "cherry"][(i % 4) as usize].to_string())
    }
}

fn with_id(name: &str, ty: DataType, nullable: bool, id: i32) -> Field {
    Field::new(name, ty, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

/// Table: id (1, int), s (2, string), later (4, string, absent from the file).
fn table_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "s", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(4, "later", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap(),
    )
}

/// Writes 30 rows in three row groups of ten; `dictionary` says whether `s` is dictionary encoded.
fn write_file(dir: &TempDir, dictionary: bool, bloom: bool) -> String {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        with_id("id", DataType::Int32, false, 1),
        with_id("s", DataType::Utf8, true, 2),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from_iter_values(0..ROWS)),
        Arc::new(StringArray::from((0..ROWS).map(value).collect::<Vec<_>>())),
    ])
    .unwrap();
    let mut props = WriterProperties::builder().set_max_row_group_row_count(Some(GROUP_ROWS));
    if !dictionary {
        props = props.set_column_dictionary_enabled(ColumnPath::from("s"), false);
    }
    if bloom {
        props = props.set_column_bloom_filter_enabled(ColumnPath::from("s"), true);
    }
    let path = format!(
        "{}/data-{dictionary}-{bloom}.parquet",
        dir.path().to_str().unwrap()
    );
    let mut writer = ArrowWriter::try_new(
        File::create(&path).unwrap(),
        arrow_schema,
        Some(props.build()),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

#[derive(Clone, Default)]
struct Read {
    dictionary_ids: Vec<i32>,
    project: Vec<i32>,
    predicate: Option<Predicate>,
    row_selection: bool,
    bloom: bool,
    batch_size: Option<usize>,
    ranges: Option<Vec<(u64, u64)>>,
    deletes: Vec<FileScanTaskDeleteFile>,
}

impl Read {
    fn new(dictionary_ids: &[i32]) -> Self {
        Self {
            dictionary_ids: dictionary_ids.to_vec(),
            project: vec![1, 2],
            ..Default::default()
        }
    }

    async fn run(&self, path: &str) -> Vec<RecordBatch> {
        let schema = table_schema();
        let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
            .with_row_selection_enabled(self.row_selection)
            .with_bloom_filter_enabled(self.bloom);
        if let Some(batch_size) = self.batch_size {
            builder = builder.with_batch_size(batch_size);
        }
        if !self.dictionary_ids.is_empty() {
            builder = builder.with_dictionary_field_ids(self.dictionary_ids.clone());
        }
        let task = FileScanTask::builder()
            .with_file_size_in_bytes(std::fs::metadata(path).unwrap().len())
            .with_start(0)
            .with_length(0)
            .with_record_count(Some(ROWS as u64))
            .with_data_file_path(path.to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema.clone())
            .with_project_field_ids(self.project.clone())
            .with_predicate(
                self.predicate
                    .clone()
                    .map(|p| p.bind(schema.clone(), true).unwrap()),
            )
            .with_row_ranges(self.ranges.clone())
            .with_deletes(self.deletes.clone())
            .with_case_sensitive(false)
            .build()
            .unwrap();
        let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
        builder
            .build()
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect()
            .await
            .unwrap()
    }
}

/// Column `index` of every batch, as strings (a dictionary is unpacked).
fn strings(batches: &[RecordBatch], index: usize) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|b| {
            let column = cast(b.column(index), &DataType::Utf8).unwrap();
            column
                .as_string::<i32>()
                .iter()
                .map(|v| v.map(ToOwned::to_owned))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn ids(batches: &[RecordBatch]) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect()
}

fn assert_dictionary(batches: &[RecordBatch], index: usize) {
    for batch in batches {
        assert_eq!(batch.column(index).data_type(), &dict_type());
        assert_eq!(batch.schema().field(index).data_type(), &dict_type());
        assert!(
            batch
                .column(index)
                .as_dictionary_opt::<Int32Type>()
                .is_some()
        );
    }
}

#[tokio::test]
async fn dictionary_encoded_column_reads_as_dictionary() {
    let dir = TempDir::new().unwrap();
    let path = write_file(&dir, true, false);

    let plain = Read::new(&[]).run(&path).await;
    assert!(
        plain
            .iter()
            .all(|b| b.column(1).data_type() == &DataType::Utf8)
    );

    let dict = Read::new(&[2]).run(&path).await;
    assert_dictionary(&dict, 1);
    assert_eq!(strings(&dict, 1), strings(&plain, 1));
    assert_eq!(strings(&dict, 1), (0..ROWS).map(value).collect::<Vec<_>>());
    // The field keeps its id.
    assert_eq!(
        dict[0]
            .schema()
            .field(1)
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY),
        Some(&"2".to_string())
    );
}

#[tokio::test]
async fn plain_encoded_column_reads_as_dictionary() {
    let dir = TempDir::new().unwrap();
    let path = write_file(&dir, false, false);
    let plain = Read::new(&[]).run(&path).await;
    let dict = Read::new(&[2]).run(&path).await;
    assert_dictionary(&dict, 1);
    assert_eq!(strings(&dict, 1), strings(&plain, 1));
}

#[tokio::test]
async fn absent_and_non_string_ids() {
    let dir = TempDir::new().unwrap();
    let path = write_file(&dir, true, false);

    // Field 4 is a string the file lacks: all-null dictionary. Field 1 is an int: unchanged.
    let mut read = Read::new(&[1, 2, 4]);
    read.project = vec![1, 2, 4];
    let dict = read.run(&path).await;
    for batch in &dict {
        assert_eq!(batch.column(0).data_type(), &DataType::Int32);
        assert_eq!(batch.column(1).data_type(), &dict_type());
        assert_eq!(batch.column(2).data_type(), &dict_type());
        assert_eq!(batch.column(2).null_count(), batch.num_rows());
    }
    assert_eq!(ids(&dict), (0..ROWS).collect::<Vec<_>>());

    // Only the absent column listed: the present one stays Utf8.
    let mut read = Read::new(&[4]);
    read.project = vec![1, 2, 4];
    let dict = read.run(&path).await;
    assert_eq!(dict[0].column(1).data_type(), &DataType::Utf8);
    assert_eq!(dict[0].column(2).data_type(), &dict_type());

    // Only a non-string id listed: nothing changes.
    let dict = Read::new(&[1]).run(&path).await;
    assert_eq!(dict[0].column(0).data_type(), &DataType::Int32);
    assert_eq!(dict[0].column(1).data_type(), &DataType::Utf8);
}

fn predicates() -> Vec<(&'static str, Predicate)> {
    let s = || Reference::new("s");
    vec![
        ("eq", s().equal_to(Datum::string("banana"))),
        ("ne", s().not_equal_to(Datum::string("banana"))),
        ("lt", s().less_than(Datum::string("b"))),
        ("le", s().less_than_or_equal_to(Datum::string("apricot"))),
        ("gt", s().greater_than(Datum::string("apple"))),
        ("ge", s().greater_than_or_equal_to(Datum::string("banana"))),
        (
            "in",
            s().is_in([Datum::string("apple"), Datum::string("cherry")]),
        ),
        (
            "not in",
            s().is_not_in([Datum::string("apple"), Datum::string("cherry")]),
        ),
        ("is null", s().is_null()),
        ("is not null", s().is_not_null()),
        ("starts with", s().starts_with(Datum::string("ap"))),
        ("not starts with", s().not_starts_with(Datum::string("ap"))),
        ("absent literal", s().equal_to(Datum::string("zebra"))),
    ]
}

#[tokio::test]
async fn predicates_agree_with_and_without_the_option() {
    for dictionary_pages in [true, false] {
        let dir = TempDir::new().unwrap();
        let path = write_file(&dir, dictionary_pages, false);
        for row_selection in [false, true] {
            for (name, predicate) in predicates() {
                let mut base = Read::new(&[]);
                base.predicate = Some(predicate.clone());
                base.row_selection = row_selection;
                let mut dict = Read::new(&[2]);
                dict.predicate = Some(predicate);
                dict.row_selection = row_selection;

                let expected = base.run(&path).await;
                let got = dict.run(&path).await;
                let case = format!(
                    "{name} (dictionary pages {dictionary_pages}, row selection {row_selection})"
                );
                assert_eq!(ids(&got), ids(&expected), "{case}");
                assert_eq!(strings(&got, 1), strings(&expected, 1), "{case}");
                assert_dictionary(&got, 1);
            }
        }
    }
}

#[tokio::test]
async fn predicate_on_a_column_that_is_not_projected() {
    let dir = TempDir::new().unwrap();
    let path = write_file(&dir, true, false);
    let mut base = Read::new(&[]);
    base.project = vec![1];
    base.predicate = Some(Reference::new("s").starts_with(Datum::string("ap")));
    let mut dict = base.clone();
    dict.dictionary_ids = vec![2];
    assert_eq!(ids(&dict.run(&path).await), ids(&base.run(&path).await));
}

#[tokio::test]
async fn bloom_filter_pruning_agrees() {
    let dir = TempDir::new().unwrap();
    let path = write_file(&dir, true, true);
    for literal in ["banana", "zebra"] {
        let mut base = Read::new(&[]);
        base.predicate = Some(Reference::new("s").equal_to(Datum::string(literal)));
        base.bloom = true;
        base.row_selection = true;
        let mut dict = base.clone();
        dict.dictionary_ids = vec![2];
        let expected = base.run(&path).await;
        let got = dict.run(&path).await;
        assert_eq!(ids(&got), ids(&expected), "{literal}");
        assert_eq!(strings(&got, 1), strings(&expected, 1), "{literal}");
    }
}

fn write_positional_delete(
    dir: &TempDir,
    data_path: &str,
    positions: &[i64],
) -> FileScanTaskDeleteFile {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        with_id("file_path", DataType::Utf8, false, 2147483546),
        with_id("pos", DataType::Int64, false, 2147483545),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(StringArray::from(vec![data_path; positions.len()])),
        Arc::new(Int64Array::from(positions.to_vec())),
    ])
    .unwrap();
    let path = format!("{}/posdel.parquet", dir.path().to_str().unwrap());
    let mut writer =
        ArrowWriter::try_new(File::create(&path).unwrap(), arrow_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    FileScanTaskDeleteFile::builder()
        .with_file_path(path.clone())
        .with_file_size_in_bytes(std::fs::metadata(&path).unwrap().len())
        .with_file_type(DataContentType::PositionDeletes)
        .with_file_format(DataFileFormat::Parquet)
        .with_partition_spec_id(0)
        .build()
        .unwrap()
}

#[tokio::test]
async fn row_ranges_and_positional_deletes_agree() {
    let dir = TempDir::new().unwrap();
    let path = write_file(&dir, true, false);
    let delete = write_positional_delete(&dir, &path, &[3, 12, 25]);
    let ranges = Some(vec![(2, 6), (9, 14), (24, 28)]);

    for (ranges, deletes) in [
        (ranges.clone(), vec![]),
        (None, vec![delete.clone()]),
        (ranges, vec![delete]),
    ] {
        for predicate in [
            None,
            Some(Reference::new("s").starts_with(Datum::string("ap"))),
        ] {
            let mut base = Read::new(&[]);
            base.ranges = ranges.clone();
            base.deletes = deletes.clone();
            base.predicate = predicate.clone();
            let mut dict = base.clone();
            dict.dictionary_ids = vec![2];
            let expected = base.run(&path).await;
            let got = dict.run(&path).await;
            assert!(!ids(&expected).is_empty());
            assert_eq!(ids(&got), ids(&expected));
            assert_eq!(strings(&got, 1), strings(&expected, 1));
            assert_dictionary(&got, 1);
        }
    }
}

#[tokio::test]
async fn small_batches_across_row_groups() {
    for dictionary_pages in [true, false] {
        let dir = TempDir::new().unwrap();
        let path = write_file(&dir, dictionary_pages, false);
        let mut base = Read::new(&[]);
        base.batch_size = Some(4);
        let mut dict = base.clone();
        dict.dictionary_ids = vec![2];
        let expected = base.run(&path).await;
        let got = dict.run(&path).await;
        assert!(got.len() > 3);
        assert_dictionary(&got, 1);
        assert_eq!(ids(&got), ids(&expected));
        assert_eq!(strings(&got, 1), strings(&expected, 1));
    }
}
