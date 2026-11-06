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

//! A shareable cache of parsed Parquet footers.
//!
//! A scan that cuts a file into several byte ranges (one [`FileScanTask`](crate::scan::FileScanTask) each) opens the
//! file once per task, and each open fetches and decodes the footer and the page indexes. The cache keeps what that
//! load produces, before any task-specific schema adaptation (name mapping, INT96 coercion, the row-number column),
//! so a later task for the same file skips both the fetch and the decode. Tasks that start together for one file
//! share a single load.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use moka::future::Cache;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;

use crate::{Error, ErrorKind, Result};

/// What identifies a footer: the file, the key it is encrypted with, and which indexes were loaded with it.
/// Data files are immutable by path, so nothing else can change what the footer holds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MetadataKey {
    path: String,
    key_metadata: Option<Vec<u8>>,
    column_index: bool,
    offset_index: bool,
}

impl MetadataKey {
    pub(crate) fn new(
        path: &str,
        key_metadata: Option<&[u8]>,
        column_index: bool,
        offset_index: bool,
    ) -> Self {
        Self {
            path: path.to_string(),
            key_metadata: key_metadata.map(<[u8]>::to_vec),
            column_index,
            offset_index,
        }
    }
}

/// Counters of a [`ParquetMetadataCache`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParquetMetadataCacheStats {
    /// Opens that found the footer cached (or waited for a load another open had started).
    pub hits: u64,
    /// Opens that loaded the footer.
    pub misses: u64,
    /// Time spent loading footers on misses (fetching the tail and parsing it), summed over opens, in nanoseconds.
    pub load_nanos: u64,
    /// Footers held now (call [`ParquetMetadataCache::sync`] first for an exact figure).
    pub entries: u64,
    /// Approximate bytes the held footers take.
    pub weighted_bytes: u64,
}

struct Inner {
    cache: Cache<MetadataKey, ArrowReaderMetadata>,
    hits: AtomicU64,
    misses: AtomicU64,
    load_nanos: AtomicU64,
}

/// A cache of parsed Parquet footers, bounded by an approximate byte weight. Cloning shares it.
///
/// Hand one to [`ArrowReaderBuilder::with_metadata_cache`](super::ArrowReaderBuilder::with_metadata_cache) to
/// use it. Eviction is moka's (TinyLFU admission, LRU eviction).
#[derive(Clone)]
pub struct ParquetMetadataCache {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for ParquetMetadataCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetMetadataCache")
            .field("stats", &self.stats())
            .finish()
    }
}

impl ParquetMetadataCache {
    /// A cache holding footers up to about `capacity_bytes` (`ParquetMetaData::memory_size`).
    pub fn new(capacity_bytes: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(capacity_bytes)
            .weigher(|_key: &MetadataKey, value: &ArrowReaderMetadata| {
                u32::try_from(value.metadata().memory_size()).unwrap_or(u32::MAX)
            })
            .build();
        Self {
            inner: Arc::new(Inner {
                cache,
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                load_nanos: AtomicU64::new(0),
            }),
        }
    }

    /// The footer for `key`, loaded with `load` if the cache lacks it. Opens that ask while a load is running wait for
    /// it rather than loading again.
    pub(crate) async fn get_or_load<F>(
        &self,
        key: MetadataKey,
        load: F,
    ) -> Result<ArrowReaderMetadata>
    where
        F: Future<Output = Result<ArrowReaderMetadata>>,
    {
        let inner = &self.inner;
        let entry = inner
            .cache
            .entry(key)
            .or_try_insert_with(async {
                let started = std::time::Instant::now();
                let loaded = load.await;
                inner
                    .load_nanos
                    .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                loaded
            })
            .await
            .map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to load Parquet metadata: {e}"),
                )
            })?;
        if entry.is_fresh() {
            inner.misses.fetch_add(1, Ordering::Relaxed);
        } else {
            inner.hits.fetch_add(1, Ordering::Relaxed);
        }
        Ok(entry.into_value())
    }

    /// Runs the cache's pending bookkeeping (evictions), so that [`stats`](Self::stats) is exact.
    pub async fn sync(&self) {
        self.inner.cache.run_pending_tasks().await;
    }

    /// The counters so far.
    pub fn stats(&self) -> ParquetMetadataCacheStats {
        ParquetMetadataCacheStats {
            hits: self.inner.hits.load(Ordering::Relaxed),
            misses: self.inner.misses.load(Ordering::Relaxed),
            load_nanos: self.inner.load_nanos.load(Ordering::Relaxed),
            entries: self.inner.cache.entry_count(),
            weighted_bytes: self.inner.cache.weighted_size(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::{ArrowReader, ParquetReadOptions};
    use crate::io::FileIO;

    fn write(dir: &TempDir, name: &str) -> (String, u64) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int64, false)
                .with_metadata([(PARQUET_FIELD_ID_META_KEY.to_string(), "1".to_string())].into()),
        ]));
        let path = dir.path().join(name).to_str().unwrap().to_string();
        let file = std::fs::File::create(&path).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(100))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
        let col: ArrayRef = Arc::new(Int64Array::from_iter_values(0..1000));
        writer
            .write(&RecordBatch::try_new(schema, vec![col]).unwrap())
            .unwrap();
        writer.close().unwrap();
        let size = std::fs::metadata(&path).unwrap().len();
        (path, size)
    }

    async fn open(
        cache: &ParquetMetadataCache,
        path: &str,
        size: u64,
        bytes: &Arc<AtomicU64>,
    ) -> ArrowReaderMetadata {
        let (_reader, metadata) = ArrowReader::open_parquet_file(
            path,
            &FileIO::new_with_fs(),
            size,
            ParquetReadOptions::builder().build(),
            bytes,
            None,
            Some(cache),
        )
        .await
        .unwrap();
        metadata
    }

    #[tokio::test]
    async fn a_second_open_of_a_file_is_a_hit_and_reads_no_bytes() {
        let dir = TempDir::new().unwrap();
        let (path, size) = write(&dir, "a.parquet");
        let cache = ParquetMetadataCache::new(1 << 20);
        let bytes = Arc::new(AtomicU64::new(0));

        let first = open(&cache, &path, size, &bytes).await;
        let after_first = bytes.load(Ordering::SeqCst);
        assert!(after_first > 0, "the first open reads the footer");
        let second = open(&cache, &path, size, &bytes).await;
        assert_eq!(
            bytes.load(Ordering::SeqCst),
            after_first,
            "a hit reads nothing"
        );

        assert!(Arc::ptr_eq(first.metadata(), second.metadata()));
        assert_eq!(first.schema(), second.schema());
        assert_eq!(first.metadata().num_row_groups(), 10);
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses), (1, 1));
        assert!(stats.load_nanos > 0);
    }

    #[tokio::test]
    async fn opens_that_start_together_share_one_load() {
        let dir = TempDir::new().unwrap();
        let (path, size) = write(&dir, "a.parquet");
        let cache = ParquetMetadataCache::new(1 << 20);
        let bytes = Arc::new(AtomicU64::new(0));
        let opens = (0..8).map(|_| open(&cache, &path, size, &bytes));
        futures::future::join_all(opens).await;
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses), (7, 1));
    }

    #[tokio::test]
    async fn footers_over_the_capacity_are_evicted() {
        let dir = TempDir::new().unwrap();
        let bytes = Arc::new(AtomicU64::new(0));
        let (first, size) = write(&dir, "0.parquet");
        let one = open(&ParquetMetadataCache::new(1 << 20), &first, size, &bytes)
            .await
            .metadata()
            .memory_size() as u64;
        // Room for about two footers.
        let cache = ParquetMetadataCache::new(one * 2 + one / 2);
        for i in 0..6 {
            let (path, size) = write(&dir, &format!("f{i}.parquet"));
            open(&cache, &path, size, &bytes).await;
            cache.sync().await;
        }
        let stats = cache.stats();
        assert_eq!(stats.misses, 6);
        assert!(stats.entries <= 2, "held {} footers", stats.entries);
        assert!(stats.weighted_bytes <= one * 2 + one / 2);
    }

    #[tokio::test]
    async fn a_missing_file_is_an_error_and_not_cached() {
        let cache = ParquetMetadataCache::new(1 << 20);
        let bytes = Arc::new(AtomicU64::new(0));
        let r = ArrowReader::open_parquet_file(
            "/no/such/file.parquet",
            &FileIO::new_with_fs(),
            10,
            ParquetReadOptions::builder().build(),
            &bytes,
            None,
            Some(&cache),
        )
        .await;
        assert!(r.is_err());
        cache.sync().await;
        assert_eq!(cache.stats().entries, 0);
    }
}
