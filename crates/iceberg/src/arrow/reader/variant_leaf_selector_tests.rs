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

//! Tests of [`VariantLeafSelector`]: a shredded Variant read by some of its leaves comes out as the file's own struct of
//! just those children, and a column the selector does not name reads (and is unshredded) as before.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::schema::types::SchemaDescriptor;
use parquet_variant_compute::{VariantArray, VariantType, json_to_variant, shred_variant};
use tempfile::TempDir;

use super::{ArrowReaderBuilder, RawVariantRead, VariantLeafSelector};
use crate::io::FileIO;
use crate::scan::{FileScanTask, FileScanTaskStream};
use crate::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, SchemaRef, Type};
use crate::{Result, Runtime};

fn with_id(field: Field, id: i32) -> Field {
    field.with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

/// Four rows of `v`, shredded on `a` (a string) and `n` (a long); `id` is 0..4.
fn write_file(dir: &str) -> String {
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        Some(r#"{"a":"x","n":1,"rest":{"k":"v"}}"#),
        Some(r#"{"a":7,"other":true}"#),
        None,
        Some(r#"{"n":3}"#),
    ]));
    let shredded = shred_variant(
        &json_to_variant(&json).unwrap(),
        &DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("n", DataType::Int64, true),
        ])),
    )
    .unwrap();
    let v: ArrayRef = Arc::new(shredded.into_inner());
    let id: ArrayRef = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
    let schema = Arc::new(ArrowSchema::new(vec![
        with_id(Field::new("id", DataType::Int32, false), 1),
        with_id(
            Field::new("v", v.data_type().clone(), true).with_extension_type(VariantType),
            2,
        ),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![id, v]).unwrap();
    let path = format!("{dir}/shredded.parquet");
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
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
                NestedField::optional(2, "v", Type::Variant(crate::spec::VariantType)).into(),
            ])
            .build()
            .unwrap(),
    )
}

fn task(path: &str, project: Vec<i32>) -> FileScanTask {
    FileScanTask::builder()
        .with_file_size_in_bytes(std::fs::metadata(path).unwrap().len())
        .with_start(0)
        .with_length(0)
        .with_data_file_path(path.to_string())
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(table_schema())
        .with_project_field_ids(project)
        .with_case_sensitive(false)
        .build()
        .unwrap()
}

/// Reads the leaves of `v` whose path is listed.
struct Leaves(Vec<&'static str>);

impl VariantLeafSelector for Leaves {
    fn select(
        &self,
        _: &FileScanTask,
        parquet_schema: &SchemaDescriptor,
        _: &ArrowSchema,
    ) -> Result<Vec<RawVariantRead>> {
        let leaves = (0..parquet_schema.num_columns())
            .filter(|i| {
                self.0
                    .contains(&parquet_schema.column(*i).path().string().as_str())
            })
            .collect();
        Ok(vec![RawVariantRead {
            field_id: 2,
            leaves,
        }])
    }
}

/// A selector that names nothing: the reader reads as it would without one.
struct Nothing;

impl VariantLeafSelector for Nothing {
    fn select(
        &self,
        _: &FileScanTask,
        _: &SchemaDescriptor,
        _: &ArrowSchema,
    ) -> Result<Vec<RawVariantRead>> {
        Ok(vec![])
    }
}

async fn read(
    task: FileScanTask,
    selector: Option<Arc<dyn VariantLeafSelector>>,
) -> (Vec<RecordBatch>, u64) {
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current());
    if let Some(selector) = selector {
        builder = builder.with_variant_leaf_selector(selector);
    }
    let tasks = Box::pin(futures::stream::iter(vec![Ok(task)])) as FileScanTaskStream;
    let scan = builder.build().read(tasks).unwrap();
    let metrics = scan.metrics().clone();
    let batches = scan.stream().try_collect().await.unwrap();
    (batches, metrics.bytes_read())
}

fn child_names(array: &dyn Array) -> Vec<String> {
    array
        .as_struct()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

#[tokio::test]
async fn a_selected_leaf_subset_comes_out_as_the_pruned_struct() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap());
    let leaves = Leaves(vec!["v.metadata", "v.typed_value.n.typed_value"]);
    let (batches, _) = read(task(&path, vec![1, 2]), Some(Arc::new(leaves))).await;
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 4);
    // `id` is read as always, `v` raw: metadata and typed_value, with only the `n` field below it.
    let v = batch.column_by_name("v").unwrap();
    assert_eq!(child_names(v), ["metadata", "typed_value"]);
    let typed = v.as_struct().column_by_name("typed_value").unwrap();
    assert_eq!(child_names(typed), ["n"]);
    let n = typed.as_struct().column_by_name("n").unwrap();
    assert_eq!(child_names(n), ["typed_value"]);
    let n = n.as_struct().column_by_name("typed_value").unwrap();
    let n = n.as_primitive::<arrow_array::types::Int64Type>();
    assert_eq!(n.iter().collect::<Vec<_>>(), [Some(1), None, None, Some(3)]);
    let ids = batch.column_by_name("id").unwrap();
    assert_eq!(
        ids.as_primitive::<arrow_array::types::Int32Type>().values(),
        &[0, 1, 2, 3]
    );
}

#[tokio::test]
async fn only_the_variant_column_is_read_when_it_is_the_whole_projection() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap());
    let leaves = Leaves(vec!["v.metadata", "v.value"]);
    let (batches, bytes) = read(task(&path, vec![2]), Some(Arc::new(leaves))).await;
    assert_eq!(batches[0].num_columns(), 1);
    assert_eq!(child_names(batches[0].column(0)), ["metadata", "value"]);
    let (_, all) = read(task(&path, vec![2]), None).await;
    assert!(bytes < all, "{bytes} should be fewer than {all}");
}

#[tokio::test]
async fn naming_no_column_reads_and_unshreds_as_before() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap());
    let (plain, _) = read(task(&path, vec![1, 2]), None).await;
    let (with_selector, _) = read(task(&path, vec![1, 2]), Some(Arc::new(Nothing))).await;
    assert_eq!(plain, with_selector);
    // The table's own `metadata` and `value`.
    let v = plain[0].column_by_name("v").unwrap();
    assert_eq!(child_names(v), ["metadata", "value"]);
    let variants = VariantArray::try_new(v.as_ref()).unwrap();
    assert_eq!(
        variants.value(0).get_object_field("a").unwrap().as_string(),
        Some("x")
    );
}

#[tokio::test]
async fn a_selection_outside_the_projection_is_an_error() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap());
    let leaves = Leaves(vec!["v.metadata"]);
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current());
    builder = builder.with_variant_leaf_selector(Arc::new(leaves));
    let tasks =
        Box::pin(futures::stream::iter(vec![Ok(task(&path, vec![1]))])) as FileScanTaskStream;
    let result: std::result::Result<Vec<RecordBatch>, _> = builder
        .build()
        .read(tasks)
        .unwrap()
        .stream()
        .try_collect()
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn naming_no_leaf_of_a_column_skips_it() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap());
    let (batches, bytes) = read(task(&path, vec![1, 2]), Some(Arc::new(Leaves(vec![])))).await;
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 2);
    // `id` is read as always; `v` is all null, and typed as the table's.
    let ids = batch.column_by_name("id").unwrap();
    assert_eq!(
        ids.as_primitive::<arrow_array::types::Int32Type>().values(),
        &[0, 1, 2, 3]
    );
    let v = batch.column_by_name("v").unwrap();
    assert_eq!(v.null_count(), 4);
    let (plain, all) = read(task(&path, vec![1, 2]), None).await;
    assert_eq!(
        v.data_type(),
        plain[0].column_by_name("v").unwrap().data_type()
    );
    assert!(bytes < all, "{bytes} should be fewer than {all}");
}

#[tokio::test]
async fn skipping_the_only_column_is_an_error() {
    let dir = TempDir::new().unwrap();
    let path = write_file(dir.path().to_str().unwrap());
    let mut builder = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current());
    builder = builder.with_variant_leaf_selector(Arc::new(Leaves(vec![])));
    let tasks =
        Box::pin(futures::stream::iter(vec![Ok(task(&path, vec![2]))])) as FileScanTaskStream;
    let result: std::result::Result<Vec<RecordBatch>, _> = builder
        .build()
        .read(tasks)
        .unwrap()
        .stream()
        .try_collect()
        .await;
    assert!(result.is_err());
}
