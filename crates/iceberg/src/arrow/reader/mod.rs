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

//! Parquet file data reader

use std::collections::HashSet;
use std::sync::Arc;

use arrow_schema::Schema as ArrowSchema;
use parquet::arrow::arrow_reader::ArrowPredicate;
use parquet::schema::types::SchemaDescriptor;

use crate::Result;
use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
use crate::io::FileIO;
use crate::runtime::Runtime;
use crate::scan::FileScanTask;
use crate::util::available_parallelism;

/// Default gap between byte ranges below which they are coalesced into a
/// single request. Matches object_store's `OBJECT_STORE_COALESCE_DEFAULT`.
const DEFAULT_RANGE_COALESCE_BYTES: u64 = 1024 * 1024;

/// Default maximum number of coalesced byte ranges fetched concurrently.
/// Matches object_store's `OBJECT_STORE_COALESCE_PARALLEL`.
const DEFAULT_RANGE_FETCH_CONCURRENCY: usize = 10;

/// Default number of bytes to prefetch when parsing Parquet footer metadata.
/// Matches DataFusion's default `ParquetOptions::metadata_size_hint`.
const DEFAULT_METADATA_SIZE_HINT: usize = 512 * 1024;

#[cfg(test)]
mod dictionary_tests;
mod file_reader;
mod metadata_cache;
mod options;
mod pipeline;
mod positional_deletes;
mod predicate_visitor;
mod projection;
mod row_filter;
#[cfg(test)]
mod row_filter_factory_tests;
mod row_lineage;
#[cfg(test)]
mod variant_leaf_selector_tests;
pub use file_reader::ArrowFileReader;
pub use metadata_cache::{ParquetMetadataCache, ParquetMetadataCacheStats};
pub(crate) use options::ParquetReadOptions;
use predicate_visitor::{CollectFieldIdVisitor, PredicateConverter};
use projection::{
    add_fallback_field_ids_to_arrow_schema, apply_name_mapping_to_arrow_schema,
    find_leaf_by_field_id,
};

/// Supplies extra row filters for each file a reader opens, on top of the one the task's predicate gives.
///
/// The reader calls [`RowFilterFactory::row_filter`] once per file task, after the file's footer is read. The
/// returned predicates are appended to the Parquet `RowFilter` (after the task predicate's, in the order given,
/// each seeing only the rows the ones before it kept, so the cheap ones belong first). Parquet then decodes the predicate's columns first and fetches and decodes the pages of the
/// other projected columns only where rows were kept, which needs the page index, so a reader with a factory
/// loads it. Rows the filter drops are not returned: a caller that cannot let the reader drop them must not
/// return a predicate that does. `_pos` and `_row_id` of the rows that remain, positional deletes and
/// [`FileScanTask::row_ranges`] are unaffected.
///
/// Returning no predicates reads the file as if there were no factory.
pub trait RowFilterFactory: Send + Sync {
    /// `parquet_schema` is the file's; `arrow_schema` is the file's Arrow schema after the reader applied the
    /// task's name mapping or fallback ids, so its fields carry `PARQUET:field_id` metadata to resolve columns
    /// by field id. A predicate's `ProjectionMask` is over `parquet_schema`'s leaves, and the batch it is
    /// evaluated on holds those leaves in file order, in the file's types (a struct with only the masked
    /// children).
    fn row_filter(
        &self,
        task: &FileScanTask,
        parquet_schema: &SchemaDescriptor,
        arrow_schema: &ArrowSchema,
    ) -> Result<Vec<Box<dyn ArrowPredicate>>>;
}

/// The leaves of one top-level Variant column to read, and to pass through as the file holds them.
#[derive(Debug, Clone)]
pub struct RawVariantRead {
    /// The Iceberg field id of the top-level Variant column.
    pub field_id: i32,
    /// Indexes into the file's Parquet leaves (`parquet_schema.column(i)`) of the column's leaves to read. Empty means the
    /// caller wants none of the column in this file: nothing of it is read and it comes out null in the table's type, as a
    /// column the file lacks does (the task must then have another column to read).
    pub leaves: Vec<usize>,
}

/// Lets the caller read part of a Variant column, as the file stores it.
///
/// A file written shredded (S3 Tables compaction does) holds a Variant as `metadata`, `value` and a `typed_value` tree
/// whose shape differs file to file; the reader normally reads every leaf and rebuilds the table's `metadata` and
/// `value` from them, which costs more than reading most of the fields. The reader calls
/// [`VariantLeafSelector::select`] once per file task, after the footer is read, and for each column it returns reads only
/// the listed leaves and hands the column over *raw*: a struct holding just those children, with the file's own types
/// (not unshredded, not cast to the table's type, so its type may differ from file to file). Columns not returned read as
/// before. A column must be a top-level field the task projects, and the leaves must belong to it.
pub trait VariantLeafSelector: Send + Sync {
    /// `parquet_schema` is the file's; `arrow_schema` is the file's Arrow schema (as given to [`RowFilterFactory`]).
    fn select(
        &self,
        task: &FileScanTask,
        parquet_schema: &SchemaDescriptor,
        arrow_schema: &ArrowSchema,
    ) -> Result<Vec<RawVariantRead>>;
}

/// Builder to create ArrowReader
pub struct ArrowReaderBuilder {
    batch_size: Option<usize>,
    file_io: FileIO,
    concurrency_limit_data_files: usize,
    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
    bloom_filter_enabled: bool,
    parquet_read_options: ParquetReadOptions,
    metadata_cache: Option<ParquetMetadataCache>,
    dictionary_field_ids: HashSet<i32>,
    row_filter_factory: Option<Arc<dyn RowFilterFactory>>,
    variant_leaf_selector: Option<Arc<dyn VariantLeafSelector>>,
    runtime: Runtime,
}

impl ArrowReaderBuilder {
    /// Create a new ArrowReaderBuilder
    pub fn new(file_io: FileIO, runtime: Runtime) -> Self {
        let num_cpus = available_parallelism().get();

        ArrowReaderBuilder {
            batch_size: None,
            file_io,
            concurrency_limit_data_files: num_cpus,
            row_group_filtering_enabled: true,
            row_selection_enabled: false,
            bloom_filter_enabled: false,
            parquet_read_options: ParquetReadOptions::builder().build(),
            metadata_cache: None,
            dictionary_field_ids: HashSet::new(),
            row_filter_factory: None,
            variant_leaf_selector: None,
            runtime,
        }
    }

    /// Sets the max number of in flight data files that are being fetched
    pub fn with_data_file_concurrency_limit(mut self, val: usize) -> Self {
        self.concurrency_limit_data_files = val;
        self
    }

    /// Sets the desired size of batches in the response
    /// to something other than the default
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Determines whether to enable row group filtering.
    pub fn with_row_group_filtering_enabled(mut self, row_group_filtering_enabled: bool) -> Self {
        self.row_group_filtering_enabled = row_group_filtering_enabled;
        self
    }

    /// Determines whether to enable row selection.
    pub fn with_row_selection_enabled(mut self, row_selection_enabled: bool) -> Self {
        self.row_selection_enabled = row_selection_enabled;
        self
    }

    /// Determines whether to enable bloom filter-based row group filtering.
    ///
    /// When enabled, if a read is performed with an equality or IN predicate,
    /// the bloom filter for relevant columns in each row group is read and
    /// checked. Row groups where the bloom filter proves the value is absent
    /// are skipped entirely.
    ///
    /// Defaults to disabled. Each bloom filter is a separate read, and they are
    /// issued serially — one round trip per relevant column per row group, before
    /// any data is read. TODO(#3191)
    pub fn with_bloom_filter_enabled(mut self, bloom_filter_enabled: bool) -> Self {
        self.bloom_filter_enabled = bloom_filter_enabled;
        self
    }

    /// Provide a hint as to the number of bytes to prefetch for parsing the Parquet metadata
    ///
    /// This hint can help reduce the number of fetch requests. For more details see the
    /// [ParquetMetaDataReader documentation](https://docs.rs/parquet/latest/parquet/file/metadata/struct.ParquetMetaDataReader.html#method.with_prefetch_hint).
    pub fn with_metadata_size_hint(mut self, metadata_size_hint: usize) -> Self {
        self.parquet_read_options.metadata_size_hint = Some(metadata_size_hint);
        self
    }

    /// Sets the gap threshold for merging nearby byte ranges into a single request.
    /// Ranges with gaps smaller than this value will be coalesced.
    ///
    /// Defaults to 1 MiB, matching object_store's OBJECT_STORE_COALESCE_DEFAULT.
    pub fn with_range_coalesce_bytes(mut self, range_coalesce_bytes: u64) -> Self {
        self.parquet_read_options.range_coalesce_bytes = range_coalesce_bytes;
        self
    }

    /// Sets the maximum number of merged byte ranges to fetch concurrently.
    ///
    /// Defaults to 10, matching object_store's OBJECT_STORE_COALESCE_PARALLEL.
    pub fn with_range_fetch_concurrency(mut self, range_fetch_concurrency: usize) -> Self {
        self.parquet_read_options.range_fetch_concurrency = range_fetch_concurrency;
        self
    }

    /// Keeps the parsed Parquet footers of the data files this reader opens in `cache`, so that several tasks on one
    /// file (byte ranges of it) fetch and decode the footer and page indexes once. Off by default. The cache can be
    /// shared by many readers. A cached footer is not re-read, so the reader's read-ahead of a file's tail (in a
    /// caching `FileIO`) does not see the tail again.
    pub fn with_metadata_cache(mut self, cache: ParquetMetadataCache) -> Self {
        self.metadata_cache = Some(cache);
        self
    }

    /// Reads the listed Iceberg field ids as `Dictionary(Int32, Utf8)` instead of `Utf8`, with the same values,
    /// when they are top-level `string` fields of a task's schema. Other ids (a field the file lacks comes out as
    /// all nulls of the dictionary type) are ignored. Off unless called. Counting a dictionary's keys is much
    /// cheaper than hashing its strings, and arrow-rs decodes dictionary-encoded Parquet pages straight into keys.
    pub fn with_dictionary_field_ids(mut self, ids: impl IntoIterator<Item = i32>) -> Self {
        self.dictionary_field_ids = ids.into_iter().collect();
        self
    }

    /// Adds a row filter per file from `factory` (see [`RowFilterFactory`]). Off unless called.
    pub fn with_row_filter_factory(mut self, factory: Arc<dyn RowFilterFactory>) -> Self {
        self.row_filter_factory = Some(factory);
        self
    }

    /// Reads only some leaves of the Variant columns `selector` names, and passes them through raw (see
    /// [`VariantLeafSelector`]). Off unless called.
    pub fn with_variant_leaf_selector(mut self, selector: Arc<dyn VariantLeafSelector>) -> Self {
        self.variant_leaf_selector = Some(selector);
        self
    }

    /// The parsed footer of the file `task` reads, loaded as a reader built from this builder loads it: through the
    /// builder's [`ParquetMetadataCache`] if it has one (so a reader that opens the file afterwards finds it parsed, and a
    /// load that is running is shared), with the builder's metadata size hint and page index settings. The result is the
    /// file's own, before the task adapts it (name mapping, coercions). For a caller that wants to know what a scan of
    /// `task` will read before it reads.
    pub async fn load_metadata(
        &self,
        task: &FileScanTask,
    ) -> Result<parquet::arrow::arrow_reader::ArrowReaderMetadata> {
        let bytes_read = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (_reader, metadata) = ArrowReader::open_parquet_file(
            task.data_file_path(),
            &self.file_io,
            task.file_size_in_bytes(),
            self.parquet_read_options.clone(),
            &bytes_read,
            task.key_metadata(),
            self.metadata_cache.as_ref(),
        )
        .await?;
        Ok(metadata)
    }

    /// Build the ArrowReader.
    pub fn build(self) -> ArrowReader {
        ArrowReader {
            batch_size: self.batch_size,
            file_io: self.file_io.clone(),
            delete_file_loader: CachingDeleteFileLoader::new(
                self.file_io.clone(),
                self.concurrency_limit_data_files,
                self.runtime.clone(),
            ),
            concurrency_limit_data_files: self.concurrency_limit_data_files,
            row_group_filtering_enabled: self.row_group_filtering_enabled,
            row_selection_enabled: self.row_selection_enabled,
            bloom_filter_enabled: self.bloom_filter_enabled,
            parquet_read_options: self.parquet_read_options,
            metadata_cache: self.metadata_cache,
            dictionary_field_ids: Arc::new(self.dictionary_field_ids),
            row_filter_factory: self.row_filter_factory,
            variant_leaf_selector: self.variant_leaf_selector,
        }
    }
}

/// Reads data from Parquet files
#[derive(Clone)]
pub struct ArrowReader {
    batch_size: Option<usize>,
    file_io: FileIO,
    delete_file_loader: CachingDeleteFileLoader,

    /// the maximum number of data files that can be fetched at the same time
    concurrency_limit_data_files: usize,

    row_group_filtering_enabled: bool,
    row_selection_enabled: bool,
    bloom_filter_enabled: bool,
    parquet_read_options: ParquetReadOptions,
    metadata_cache: Option<ParquetMetadataCache>,
    dictionary_field_ids: Arc<HashSet<i32>>,
    row_filter_factory: Option<Arc<dyn RowFilterFactory>>,
    variant_leaf_selector: Option<Arc<dyn VariantLeafSelector>>,
}
