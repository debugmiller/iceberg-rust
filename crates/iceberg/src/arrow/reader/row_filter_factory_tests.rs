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

//! Tests of [`RowFilterFactory`]: the extra row filter keeps the rows the filter keeps and nothing else,
//! leaves `_pos`, deletes and row ranges right, and lets the reader skip pages of the other columns.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::arrow_reader::{ArrowPredicate, ArrowPredicateFn};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY, ProjectionMask};
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::SchemaDescriptor;
use tempfile::TempDir;

use super::{ArrowReaderBuilder, RowFilterFactory, find_leaf_by_field_id};
use crate::io::FileIO;
use crate::metadata_columns::{RESERVED_COL_NAME_POS, RESERVED_FIELD_ID_POS};
use crate::scan::{FileScanTask, FileScanTaskDeleteFile, FileScanTaskStream};
use crate::spec::{
    DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema, SchemaRef, Type,
};
use crate::{Result, Runtime};

const ROWS: i32 = 40_000;
const PAGE_ROWS: usize = 1_000;

/// Keeps the rows whose `id` (field id 1) is below `ids_per_period` in a period of 10 000.
struct IdPeriod {
    ids_per_period: i32,
}

impl RowFilterFactory for IdPeriod {
    fn row_filter(
        &self,
        _task: &FileScanTask,
        parquet_schema: &SchemaDescriptor,
        _arrow_schema: &ArrowSchema,
    ) -> Result<Option<Box<dyn ArrowPredicate>>> {
        let leaf = find_leaf_by_field_id(parquet_schema, 1).expect("id leaf");
        let n = self.ids_per_period;
        Ok(Some(Box::new(ArrowPredicateFn::new(
            ProjectionMask::leaves(parquet_schema, [leaf]),
            move |batch: RecordBatch| {
                let ids = batch
                    .column(0)
                    .as_primitive::<arrow_array::types::Int32Type>();
                Ok(BooleanArray::from_iter(
                    ids.iter().map(|v| Some(v.is_some_and(|v| v % 10_000 < n))),
                ))
            },
        ))))
    }
}

struct NoFilter;

impl RowFilterFactory for NoFilter {
    fn row_filter(
        &self,
        _: &FileScanTask,
        _: &SchemaDescriptor,
        _: &ArrowSchema,
    ) -> Result<Option<Box<dyn ArrowPredicate>>> {
        Ok(None)
    }
}

fn field(name: &str, ty: DataType, id: i32) -> Field {
    Field::new(name, ty, false).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

/// `id` = 0..ROWS and a `wide` string of 200 varied bytes per row, in pages of `PAGE_ROWS` rows, one row group.
fn write_file(dir: &str, name: &str) -> String {
    let schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, 1),
        field("wide", DataType::Utf8, 2),
    ]));
    let ids: ArrayRef = Arc::new(Int32Array::from_iter_values(0..ROWS));
    // A cheap xorshift keeps the text incompressible, so the bytes read show which pages were fetched.
    let mut x: u64 = 0x9E3779B97F4A7C15;
    let wide: Vec<String> = (0..ROWS)
        .map(|_| {
            (0..200)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    char::from(b'a' + (x % 26) as u8)
                })
                .collect()
        })
        .collect();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![ids, Arc::new(StringArray::from(wide))]).unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .set_max_row_group_row_count(Some(ROWS as usize))
        .build();
    let path = format!("{dir}/{name}");
    let mut writer =
        ArrowWriter::try_new(File::create(&path).unwrap(), schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

fn table_schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "wide", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap(),
    )
}

fn task(
    path: &str,
    project: Vec<i32>,
    deletes: Vec<FileScanTaskDeleteFile>,
    row_ranges: Option<Vec<(u64, u64)>>,
) -> FileScanTask {
    FileScanTask::builder()
        .with_file_size_in_bytes(std::fs::metadata(path).unwrap().len())
        .with_start(0)
        .with_length(0)
        .with_data_file_path(path.to_string())
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(table_schema())
        .with_project_field_ids(project)
        .with_deletes(deletes)
        .with_row_ranges(row_ranges)
        .with_case_sensitive(false)
        .build()
        .unwrap()
}

async fn read(
    task: FileScanTask,
    factory: Option<Arc<dyn RowFilterFactory>>,
) -> (Vec<RecordBatch>, u64) {
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current());
    if let Some(factory) = factory {
        builder = builder.with_row_filter_factory(factory);
    }
    let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
    let scan = builder.build().read(tasks).unwrap();
    let metrics = scan.metrics().clone();
    let batches = scan.stream().try_collect().await.unwrap();
    (batches, metrics.bytes_read())
}

fn ids(batches: &[RecordBatch]) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|b| {
            b.column_by_name("id")
                .unwrap()
                .as_primitive::<arrow_array::types::Int32Type>()
                .values()
                .to_vec()
        })
        .collect()
}

fn positions(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            b.column_by_name(RESERVED_COL_NAME_POS)
                .unwrap()
                .as_primitive::<arrow_array::types::Int64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

fn wides(batches: &[RecordBatch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b.column_by_name("wide").unwrap().as_string::<i32>();
            (0..a.len())
                .map(|i| a.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn factory(n: i32) -> Option<Arc<dyn RowFilterFactory>> {
    Some(Arc::new(IdPeriod { ids_per_period: n }))
}

fn kept(id: i32) -> bool {
    id % 10_000 < 3
}

#[tokio::test]
async fn rows_equal_the_unfiltered_rows_the_filter_keeps() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap(), "f.parquet");
    let project = vec![1, 2, RESERVED_FIELD_ID_POS];

    let (all, _) = read(task(&path, project.clone(), vec![], None), None).await;
    let (filtered, _) = read(task(&path, project.clone(), vec![], None), factory(3)).await;

    let all_ids = ids(&all);
    let all_pos = positions(&all);
    let all_wide = wides(&all);
    let expect: Vec<usize> = (0..all_ids.len()).filter(|&i| kept(all_ids[i])).collect();
    assert_eq!(expect.len(), 12);
    assert_eq!(
        ids(&filtered),
        expect.iter().map(|&i| all_ids[i]).collect::<Vec<_>>()
    );
    // `_pos` keeps the absolute position of a surviving row.
    assert_eq!(
        positions(&filtered),
        expect.iter().map(|&i| all_pos[i]).collect::<Vec<_>>()
    );
    assert_eq!(
        wides(&filtered),
        expect
            .iter()
            .map(|&i| all_wide[i].clone())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn a_factory_that_returns_none_reads_as_without_one() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap(), "f.parquet");
    let (a, _) = read(task(&path, vec![1, 2], vec![], None), None).await;
    let (b, _) = read(
        task(&path, vec![1, 2], vec![], None),
        Some(Arc::new(NoFilter)),
    )
    .await;
    assert_eq!(ids(&a), ids(&b));
    assert_eq!(wides(&a), wides(&b));
    assert_eq!(ids(&b).len(), ROWS as usize);
}

#[tokio::test]
async fn a_selective_filter_skips_the_pages_of_the_other_column() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap(), "f.parquet");
    let (all, without) = read(task(&path, vec![1, 2], vec![], None), None).await;
    let (filtered, with) = read(task(&path, vec![1, 2], vec![], None), factory(3)).await;
    assert_eq!(ids(&filtered).len(), 12);
    assert_eq!(ids(&all).len(), ROWS as usize);
    // 4 of 40 pages of `wide` hold a kept row, and the `id` column is read whole by the filter.
    assert!(
        with * 5 < without,
        "bytes with the filter {with}, without {without}"
    );
}

#[tokio::test]
async fn pos_only_and_count_projections_follow_the_filter() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap(), "f.parquet");

    let (pos, _) = read(
        task(&path, vec![RESERVED_FIELD_ID_POS], vec![], None),
        factory(3),
    )
    .await;
    let want: Vec<i64> = (0..ROWS as i64).filter(|&p| kept(p as i32)).collect();
    assert_eq!(positions(&pos), want);

    let (count, _) = read(task(&path, vec![], vec![], None), factory(3)).await;
    assert_eq!(
        count.iter().map(|b| b.num_rows()).sum::<usize>(),
        want.len()
    );
}

fn write_positional_delete(dir: &str, data_path: &str, pos: &[i64]) -> FileScanTaskDeleteFile {
    let schema = Arc::new(ArrowSchema::new(vec![
        field("file_path", DataType::Utf8, 2147483546),
        field("pos", DataType::Int64, 2147483545),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(StringArray::from(vec![data_path; pos.len()])),
        Arc::new(Int64Array::from(pos.to_vec())),
    ])
    .unwrap();
    let path = format!("{dir}/deletes.parquet");
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
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
async fn deletes_and_row_ranges_still_apply_under_the_filter() {
    let dir = TempDir::new().unwrap();
    let d = dir.path().to_str().unwrap();
    let path = write_file(d, "f.parquet");
    // Kept by the filter: 0..3, 10000..10003, 20000..20003, 30000..30003.
    let delete = write_positional_delete(d, &path, &[1, 10_001, 5]);
    let project = vec![1, 2, RESERVED_FIELD_ID_POS];

    let (batches, _) = read(
        task(&path, project.clone(), vec![delete.clone()], None),
        factory(3),
    )
    .await;
    assert_eq!(ids(&batches), vec![
        0, 2, 10_000, 10_002, 20_000, 20_001, 20_002, 30_000, 30_001, 30_002
    ]);
    assert_eq!(
        positions(&batches),
        ids(&batches).iter().map(|&i| i as i64).collect::<Vec<_>>()
    );

    // Ranges [1, 10001) and [20001, 30002): the row ranges, deletes and the filter all have to keep a row.
    let ranges = Some(vec![(1u64, 10_001u64), (20_001, 30_002)]);
    let (batches, _) = read(task(&path, project, vec![delete], ranges), factory(3)).await;
    assert_eq!(ids(&batches), vec![
        2, 10_000, 20_001, 20_002, 30_000, 30_001
    ]);
    // The wide column of those rows is the file's own.
    let (all, _) = read(task(&path, vec![1, 2], vec![], None), None).await;
    let (all_ids, all_wide) = (ids(&all), wides(&all));
    let want: Vec<String> = ids(&batches)
        .iter()
        .map(|id| all_wide[all_ids.iter().position(|x| x == id).unwrap()].clone())
        .collect();
    assert_eq!(wides(&batches), want);
}
