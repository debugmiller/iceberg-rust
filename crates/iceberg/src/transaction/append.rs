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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{DataFile, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, MANIFEST_MERGE_TARGET_SIZE_BYTES, MergeManifestProcess,
    SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// A check run against the table the append is about to be applied to.
type Precondition = Arc<dyn Fn(&Table) -> Result<()> + Send + Sync>;

/// FastAppendAction is a transaction action for fast append data files to the table.
pub struct FastAppendAction {
    check_duplicate: bool,
    precondition: Option<Precondition>,
    // below are properties used to create SnapshotProducer when commit
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    merge_manifests_from: Option<usize>,
}

impl FastAppendAction {
    pub(crate) fn new() -> Self {
        Self {
            check_duplicate: true,
            precondition: None,
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            merge_manifests_from: None,
        }
    }

    /// Rewrite the small data manifests the snapshot carries forward as fewer, larger ones once
    /// `min_count` of them have built up (see `MergeManifestProcess`), instead of leaving one manifest
    /// per commit. The manifest this commit adds is not merged until a later commit.
    pub fn with_manifest_merge(mut self, min_count: usize) -> Self {
        self.merge_manifests_from = Some(min_count);
        self
    }

    /// Set whether to check duplicate files
    pub fn with_check_duplicate(mut self, v: bool) -> Self {
        self.check_duplicate = v;
        self
    }

    /// Run `check` against the table every time the append is applied: first on the table the transaction
    /// started from, then on the refreshed table each time a commit conflict makes it retry. An error from it
    /// stops the commit and is returned as it is; mark it non-retryable (the default), or the commit retries.
    ///
    /// This is how a caller makes "has this already been committed?" atomic with the commit: the check sees
    /// exactly the table the commit is based on, and a concurrent commit that changes it fails this commit's
    /// compare-and-swap, which re-runs the check.
    pub fn with_precondition(
        mut self,
        check: impl Fn(&Table) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.precondition = Some(Arc::new(check));
        self
    }

    /// Add data files to the snapshot.
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Set commit UUID for the snapshot.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }

    /// Collapse files sharing a path to their first occurrence, so a single
    /// manifest never references the same file twice. Always runs (unlike the
    /// `check_duplicate`-gated cross-snapshot check) since it is in-memory only.
    fn dedupe_added_files(&self) -> Vec<DataFile> {
        let mut seen = HashSet::with_capacity(self.added_data_files.len());
        self.added_data_files
            .iter()
            .filter(|data_file| seen.insert(data_file.file_path.as_str()))
            .cloned()
            .collect()
    }
}

#[async_trait]
impl TransactionAction for FastAppendAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if let Some(check) = &self.precondition {
            check(table)?;
        }
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.snapshot_properties.clone(),
            self.dedupe_added_files(),
        );

        // validate added files
        snapshot_producer.validate_added_data_files()?;

        // Checks duplicate files
        if self.check_duplicate {
            snapshot_producer.validate_duplicate_files().await?;
        }

        match self.merge_manifests_from {
            Some(min_count) => {
                snapshot_producer
                    .commit(FastAppendOperation, MergeManifestProcess {
                        min_count,
                        target_size_bytes: MANIFEST_MERGE_TARGET_SIZE_BYTES,
                    })
                    .await
            }
            None => {
                snapshot_producer
                    .commit(FastAppendOperation, DefaultManifestProcess)
                    .await
            }
        }
    }
}

struct FastAppendOperation;

impl SnapshotProduceOperation for FastAppendOperation {
    fn operation(&self) -> Operation {
        Operation::Append
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let Some(snapshot) = snapshot_produce.table.metadata().current_snapshot() else {
            return Ok(vec![]);
        };

        let manifest_list = snapshot_produce
            .table
            .manifest_list_reader(snapshot)
            .load()
            .await?;

        Ok(manifest_list
            .entries()
            .iter()
            .filter(|entry| {
                // Keep delete-only manifests too: they record which files were removed and
                // must persist across snapshots until `expire_snapshots` cleans them up.
                // Dropping them lets the removed files reappear as live data (see #2148).
                entry.has_added_files() || entry.has_existing_files() || entry.has_deleted_files()
            })
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::sync::Arc;

    use minijinja::{AutoEscape, Environment, Value, context};
    use tempfile::TempDir;
    use uuid::Uuid;

    use crate::encryption::kms::MemoryKeyManagementClient;
    use crate::encryption::{SensitiveBytes, StandardKeyMetadata};
    use crate::io::FileIO;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, MAIN_BRANCH,
        ManifestEntry, ManifestListWriter, ManifestStatus, ManifestWriterBuilder, SnapshotRef,
        Struct, TableMetadata,
    };
    use crate::table::Table;
    use crate::test_utils::{make_encrypted_table, test_runtime};
    use crate::transaction::tests::make_v2_minimal_table;
    use crate::transaction::{Transaction, TransactionAction};
    use crate::{TableIdent, TableRequirement, TableUpdate};

    fn render_template(template: &str, ctx: Value) -> String {
        let mut env = Environment::new();
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.render_str(template, ctx).unwrap()
    }

    /// Builds a table whose current snapshot's manifest list contains a data manifest
    /// followed by a delete-only manifest (one entry with `ManifestStatus::Deleted`,
    /// so `deleted_files_count > 0` while `added_files_count == existing_files_count == 0`).
    ///
    /// Returns the table plus the `manifest_path` of the delete-only manifest so callers
    /// can assert whether a subsequent append carries it forward.
    async fn make_table_with_delete_only_manifest() -> (Table, TempDir, String) {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().join("table1");
        let manifest_list_location = table_location.join("metadata/manifests_list_1.avro");
        let table_metadata_location = table_location.join("metadata/v1.json");

        let file_io = FileIO::new_with_fs();

        let template = fs::read_to_string(format!(
            "{}/testdata/example_table_metadata_v2.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        // The template has two snapshots; point the current one at our manifest list.
        let metadata_json = render_template(&template, context! {
            table_location => &table_location,
            manifest_list_1_location => &manifest_list_location,
            manifest_list_2_location => &manifest_list_location,
            table_metadata_1_location => &table_metadata_location,
        });
        let table_metadata = serde_json::from_str::<TableMetadata>(&metadata_json).unwrap();

        let table = Table::builder()
            .metadata(table_metadata)
            .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
            .file_io(file_io)
            .metadata_location(table_metadata_location.to_str().unwrap())
            .runtime(test_runtime())
            .build()
            .unwrap();

        let current_snapshot = table.metadata().current_snapshot().unwrap();
        let schema = current_snapshot.schema(table.metadata()).unwrap();
        let partition_spec = table.metadata().default_partition_spec();

        let next_manifest_file = |location: &str| {
            table
                .file_io()
                .new_output(format!(
                    "{}/metadata/manifest_{}.avro",
                    location,
                    Uuid::new_v4()
                ))
                .unwrap()
        };
        let table_location_str = table_location.to_str().unwrap().to_string();

        // Data manifest: one Added data file.
        let mut data_writer = ManifestWriterBuilder::new(
            next_manifest_file(&table_location_str),
            Some(current_snapshot.snapshot_id()),
            schema.clone(),
            partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        data_writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{table_location_str}/data.parquet"))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(100)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let data_manifest = data_writer.write_manifest_file().await.unwrap();

        // Delete-only manifest: a single Deleted entry, nothing added or existing.
        let mut delete_writer = ManifestWriterBuilder::new(
            next_manifest_file(&table_location_str),
            Some(current_snapshot.snapshot_id()),
            schema.clone(),
            partition_spec.as_ref().clone(),
        )
        .build_v2_data();
        delete_writer
            .add_delete_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Deleted)
                    .sequence_number(0)
                    .file_sequence_number(0)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{table_location_str}/removed.parquet"))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(100)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();
        let delete_manifest = delete_writer.write_manifest_file().await.unwrap();
        let delete_manifest_path = delete_manifest.manifest_path.clone();

        // Sanity: the delete manifest really is delete-only.
        assert!(delete_manifest.has_deleted_files());
        assert!(!delete_manifest.has_added_files());
        assert!(!delete_manifest.has_existing_files());

        let mut manifest_list_writer = ManifestListWriter::v2(
            table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .unwrap()
                .writer()
                .await
                .unwrap(),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_writer
            .add_manifests(vec![data_manifest, delete_manifest].into_iter())
            .unwrap();
        manifest_list_writer.close().await.unwrap();

        (table, tmp_dir, delete_manifest_path)
    }

    /// Regression test for #2148: a `fast_append` must carry delete-only manifests
    /// forward into the new snapshot. Dropping them lets the files they mark as
    /// removed reappear as live data on the next append.
    #[tokio::test]
    async fn test_fast_append_preserves_delete_only_manifest() {
        let (table, _tmp_dir, delete_manifest_path) = make_table_with_delete_only_manifest().await;

        // Append a new data file via the public transaction API.
        let new_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("{}/appended.parquet", table.metadata().location()))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .build()
            .unwrap();

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![new_file]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        let new_snapshot: SnapshotRef = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            SnapshotRef::new(snapshot.clone())
        } else {
            unreachable!("first update of a fast append should be AddSnapshot")
        };

        let manifest_list = table
            .manifest_list_reader(&new_snapshot)
            .load()
            .await
            .unwrap();

        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|m| m.manifest_path == delete_manifest_path),
            "delete-only manifest {delete_manifest_path} was dropped from the new snapshot's \
             manifest list; the files it removed would reappear as live data"
        );
    }

    /// Load the data files written by a single-manifest fast-append commit.
    async fn committed_data_files(table: &Table, updates: &[TableUpdate]) -> Vec<DataFile> {
        let TableUpdate::AddSnapshot { snapshot } = &updates[0] else {
            unreachable!("first update is always AddSnapshot")
        };
        let manifest_list = table
            .manifest_list_reader(&SnapshotRef::new(snapshot.clone()))
            .load()
            .await
            .unwrap();
        assert_eq!(1, manifest_list.entries().len());
        table
            .manifest_reader()
            .read(&manifest_list.entries()[0])
            .await
            .unwrap()
            .entries()
            .iter()
            .map(|entry| entry.data_file().clone())
            .collect()
    }

    #[tokio::test]
    async fn test_fast_append_writes_encrypted_manifest() {
        let table = make_encrypted_table().await;
        assert!(
            table.encryption_manager().is_some(),
            "fixture table should have an EncryptionManager"
        );

        let new_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("memory:///table/data/00000.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .partition(Struct::empty())
            .record_count(100)
            .file_size_in_bytes(4096)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .build()
            .unwrap();

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![new_file]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        let new_snapshot: SnapshotRef = updates
            .iter()
            .find_map(|u| match u {
                TableUpdate::AddSnapshot { snapshot } => Some(SnapshotRef::new(snapshot.clone())),
                _ => None,
            })
            .expect("a fast append should emit an AddSnapshot update");

        let manifest_list_key_metadata = table
            .encryption_manager()
            .unwrap()
            .decrypt_manifest_list_key_metadata(new_snapshot.encryption_key_id().unwrap())
            .await
            .unwrap();
        let manifest_list_size = table
            .file_io()
            .new_input(new_snapshot.manifest_list())
            .unwrap()
            .metadata()
            .await
            .unwrap()
            .size;
        assert_eq!(
            manifest_list_key_metadata.file_length(),
            Some(manifest_list_size)
        );

        let manifest_list = table
            .manifest_list_reader(&new_snapshot)
            .load()
            .await
            .unwrap();
        let manifest_file = manifest_list
            .entries()
            .iter()
            .find(|m| m.added_files_count.unwrap_or(0) > 0)
            .expect("new snapshot should carry the appended data manifest");

        // The manifest list entry must carry decodable key metadata.
        let key_metadata_bytes = manifest_file
            .key_metadata
            .as_ref()
            .expect("encrypted manifest must record key metadata");
        let key_metadata = StandardKeyMetadata::decode(key_metadata_bytes)
            .expect("recorded key metadata must decode as StandardKeyMetadata");
        let manifest_size = table
            .file_io()
            .new_input(&manifest_file.manifest_path)
            .unwrap()
            .metadata()
            .await
            .unwrap()
            .size;
        assert_eq!(key_metadata.file_length(), Some(manifest_size));
        assert_eq!(manifest_file.manifest_length, manifest_size as i64);

        // The reader self-decrypts using the recorded key metadata and must
        // recover the entry we appended. Because the read goes through the
        // decryption path, this succeeding also proves the bytes
        // on disk were genuinely encrypted (not silently written as plaintext).
        let manifest = table.manifest_reader().read(manifest_file).await.unwrap();
        assert_eq!(manifest.entries().len(), 1);
        assert_eq!(
            manifest.entries()[0].data_file().file_path(),
            "memory:///table/data/00000.parquet"
        );
    }

    /// The precondition sees the table the append is applied to and stops the commit when it fails.
    #[tokio::test]
    async fn test_precondition_stops_the_append() {
        use crate::error::{Error, ErrorKind};

        let table = make_v2_minimal_table();
        let file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/precondition.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .build()
            .unwrap();
        let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = seen.clone();
        let stopped = Transaction::new(&table)
            .fast_append()
            .add_data_files(vec![file.clone()])
            .with_precondition(move |_| {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err(Error::new(ErrorKind::PreconditionFailed, "already there"))
            });
        let err = Arc::new(stopped)
            .commit(&table)
            .await
            .err()
            .expect("the check failed");
        assert_eq!(err.kind(), ErrorKind::PreconditionFailed);
        assert!(!err.retryable());
        assert_eq!(seen.load(std::sync::atomic::Ordering::SeqCst), 1);

        let passed = Transaction::new(&table)
            .fast_append()
            .add_data_files(vec![file])
            .with_precondition(|_| Ok(()));
        assert!(Arc::new(passed).commit(&table).await.is_ok());
    }

    #[tokio::test]
    async fn test_empty_data_append_action() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![]);
        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    /// A `fast_append` must write the manifest list and the manifest
    /// files under the `write.metadata.path` prefix when configured,
    /// rather than the default `<location>/metadata` directory.
    #[tokio::test]
    async fn test_fast_append_honors_write_metadata_path() {
        let base = make_v2_minimal_table();
        let metadata_root = format!("{}/custom-meta", base.metadata().location());
        let metadata = base
            .metadata()
            .clone()
            .into_builder(None)
            .set_properties(HashMap::from([(
                "write.metadata.path".to_string(),
                metadata_root.clone(),
            )]))
            .unwrap()
            .build()
            .unwrap()
            .metadata;
        let table = base.with_metadata(Arc::new(metadata));

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("{}/data/1.parquet", table.metadata().location()))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .build()
            .unwrap();

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![data_file]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        let new_snapshot: SnapshotRef = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            SnapshotRef::new(snapshot.clone())
        } else {
            unreachable!("first update of a fast append should be AddSnapshot")
        };

        let prefix = format!("{metadata_root}/");

        // Manifest list
        assert!(
            new_snapshot.manifest_list().starts_with(prefix.as_str()),
            "manifest list {} not under configured write.metadata.path {metadata_root}",
            new_snapshot.manifest_list()
        );

        // Manifest files
        let manifest_list = table
            .manifest_list_reader(&new_snapshot)
            .load()
            .await
            .unwrap();
        assert!(
            !manifest_list.entries().is_empty(),
            "expected at least one manifest entry"
        );
        for entry in manifest_list.entries() {
            assert!(
                entry.manifest_path.starts_with(prefix.as_str()),
                "manifest {} not under configured write.metadata.path {metadata_root}",
                entry.manifest_path
            );
        }
    }

    #[tokio::test]
    async fn test_set_snapshot_properties() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let mut snapshot_properties = HashMap::new();
        snapshot_properties.insert("key".to_string(), "val".to_string());

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/1.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .build()
            .unwrap();

        let action = tx
            .fast_append()
            .set_snapshot_properties(snapshot_properties)
            .add_data_files(vec![data_file]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        // Check customized properties is contained in snapshot summary properties.
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };
        assert_eq!(
            new_snapshot
                .summary()
                .additional_properties
                .get("key")
                .unwrap(),
            "val"
        );
    }

    /// See `testdata/manifests_lists/README.md`.
    const FIXTURE_MASTER_KEY_ID: &str = "master-1";
    const FIXTURE_MASTER_KEY_BYTES: [u8; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f,
    ];

    async fn make_v3_encrypted_table() -> Table {
        let json = fs::read_to_string(format!(
            "{}/testdata/table_metadata/TableMetadataV3ValidEncryption.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let metadata = serde_json::from_str::<TableMetadata>(&json).unwrap();

        let kms = MemoryKeyManagementClient::new();
        kms.add_master_key_bytes(
            FIXTURE_MASTER_KEY_ID,
            SensitiveBytes::new(FIXTURE_MASTER_KEY_BYTES),
        )
        .unwrap();

        let file_io = FileIO::new_with_memory();

        let manifest_list_bytes = fs::read(format!(
            "{}/testdata/manifests_lists/manifest-list-v3-encrypted.avro",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let parent_manifest_list = metadata.current_snapshot().unwrap().manifest_list();
        file_io
            .new_output(parent_manifest_list)
            .unwrap()
            .write(manifest_list_bytes.into())
            .await
            .unwrap();

        Table::builder()
            .metadata(metadata)
            .metadata_location("memory:///table/metadata/v1.json")
            .identifier(TableIdent::from_strs(["ns1", "enc"]).unwrap())
            .file_io(file_io)
            .kms_client(Arc::new(kms))
            .runtime(test_runtime())
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_commit_with_encryption_adds_keys_and_records_snapshot_key_id() {
        let table = make_v3_encrypted_table().await;

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/1.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::empty())
            .build()
            .unwrap();

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![data_file]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        let added_key_ids: Vec<String> = updates
            .iter()
            .filter_map(|u| match u {
                TableUpdate::AddEncryptionKey { encryption_key } => {
                    Some(encryption_key.key_id().to_string())
                }
                _ => None,
            })
            .collect();
        assert!(!added_key_ids.is_empty(), "got {updates:?}");

        // Encryption keys are added before the snapshot, so it isn't updates[0] here.
        let new_snapshot = updates
            .iter()
            .find_map(|u| match u {
                TableUpdate::AddSnapshot { snapshot } => Some(snapshot),
                _ => None,
            })
            .expect("commit should add a snapshot");

        let snapshot_key_id = new_snapshot
            .encryption_key_id()
            .expect("encrypted snapshot should record its manifest-list key id");
        assert!(
            added_key_ids.iter().any(|id| id == snapshot_key_id),
            "snapshot key id {snapshot_key_id} not in added keys {added_key_ids:?}"
        );

        let new_snapshot_ref: SnapshotRef = Arc::new(new_snapshot.clone());
        let manifest_list = table
            .manifest_list_reader(&new_snapshot_ref)
            .load()
            .await
            .expect("newly written encrypted manifest list should decrypt and parse");
        assert_eq!(
            manifest_list.entries().len(),
            1,
            "append should record exactly the one new data manifest"
        );
    }

    #[tokio::test]
    async fn test_snapshot_properties_cannot_override_computed_metrics() {
        // A user-supplied snapshot property must not shadow a computed metric key
        // such as `added-data-files`. Matching iceberg-java, the computed value
        // wins, so the summary reflects the real count and a bad value can neither
        // corrupt the summary nor panic total computation (see #2184-adjacent fix).
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let mut snapshot_properties = HashMap::new();
        // Both a benign-but-wrong value and a non-integer value collide with
        // computed metric keys; neither should reach the final summary.
        snapshot_properties.insert("added-data-files".to_string(), "9999".to_string());
        snapshot_properties.insert("added-records".to_string(), "not-a-number".to_string());

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/1.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .build()
            .unwrap();

        let action = tx
            .fast_append()
            .set_snapshot_properties(snapshot_properties)
            .add_data_files(vec![data_file]);
        // Must not panic during total computation.
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };
        let props = &new_snapshot.summary().additional_properties;

        // Computed metric wins over the user's colliding values.
        assert_eq!(
            props.get("added-data-files").unwrap(),
            "1",
            "computed added-data-files must override the user-supplied value"
        );
        assert_eq!(
            props.get("added-records").unwrap(),
            "1",
            "computed added-records must override the user-supplied non-integer value"
        );
    }

    #[tokio::test]
    async fn test_append_snapshot_properties() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let mut snapshot_properties = HashMap::new();
        snapshot_properties.insert("key".to_string(), "val".to_string());

        let action = tx
            .fast_append()
            .set_snapshot_properties(snapshot_properties);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        // Check customized properties is contained in snapshot summary properties.
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };
        assert_eq!(
            new_snapshot
                .summary()
                .additional_properties
                .get("key")
                .unwrap(),
            "val"
        );
    }

    #[tokio::test]
    async fn test_fast_append_file_with_incompatible_partition_value() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);
        let action = tx.fast_append();

        // check add data file with incompatible partition value
        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/3.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::string("test"))]))
            .build()
            .unwrap();

        let action = action.add_data_files(vec![data_file.clone()]);

        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    #[tokio::test]
    async fn test_fast_append_dedupes_intra_batch_duplicate_paths() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let make_file = |size: u64, records: u64| {
            DataFileBuilder::default()
                .content(DataContentType::Data)
                .file_path("test/dup.parquet".to_string())
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(size)
                .record_count(records)
                .partition_spec_id(table.metadata().default_partition_spec_id())
                .partition(Struct::from_iter([Some(Literal::long(1))]))
                .build()
                .unwrap()
        };

        // Same path three times: the manifest keeps a single entry, the first one.
        let action = tx.fast_append().add_data_files(vec![
            make_file(100, 10),
            make_file(200, 20),
            make_file(300, 30),
        ]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let files = committed_data_files(&table, &action_commit.take_updates()).await;
        assert_eq!(1, files.len());
        assert_eq!(100, files[0].file_size_in_bytes());
    }

    #[tokio::test]
    async fn test_fast_append_dedupes_regardless_of_check_duplicate_flag() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let make_file = || {
            DataFileBuilder::default()
                .content(DataContentType::Data)
                .file_path("test/dup.parquet".to_string())
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(100)
                .record_count(10)
                .partition_spec_id(table.metadata().default_partition_spec_id())
                .partition(Struct::from_iter([Some(Literal::long(1))]))
                .build()
                .unwrap()
        };

        // `check_duplicate` only gates the cross-snapshot check; intra-batch dedupe runs regardless.
        let action = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![make_file(), make_file()]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let files = committed_data_files(&table, &action_commit.take_updates()).await;
        assert_eq!(1, files.len());
    }

    #[tokio::test]
    async fn test_fast_append() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);
        let action = tx.fast_append();

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/3.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .build()
            .unwrap();

        let action = action.add_data_files(vec![data_file.clone()]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();
        let requirements = action_commit.take_requirements();

        // check updates and requirements
        assert!(
            matches!((&updates[0],&updates[1]), (TableUpdate::AddSnapshot { snapshot },TableUpdate::SetSnapshotRef { reference,ref_name }) if snapshot.snapshot_id() == reference.snapshot_id && ref_name == MAIN_BRANCH)
        );
        assert_eq!(
            vec![
                TableRequirement::UuidMatch {
                    uuid: table.metadata().uuid()
                },
                TableRequirement::RefSnapshotIdMatch {
                    r#ref: MAIN_BRANCH.to_string(),
                    snapshot_id: table.metadata().current_snapshot_id
                }
            ],
            requirements
        );

        // check manifest list
        let new_snapshot: SnapshotRef = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            SnapshotRef::new(snapshot.clone())
        } else {
            unreachable!()
        };
        let manifest_list = table
            .manifest_list_reader(&new_snapshot)
            .load()
            .await
            .unwrap();
        assert_eq!(1, manifest_list.entries().len());
        assert_eq!(
            manifest_list.entries()[0].sequence_number,
            new_snapshot.sequence_number()
        );

        // check manifest
        let manifest = table
            .manifest_reader()
            .read(&manifest_list.entries()[0])
            .await
            .unwrap();
        assert_eq!(1, manifest.entries().len());
        assert_eq!(
            new_snapshot.sequence_number(),
            manifest.entries()[0]
                .sequence_number()
                .expect("Inherit sequence number by load manifest")
        );

        assert_eq!(
            new_snapshot.snapshot_id(),
            manifest.entries()[0].snapshot_id().unwrap()
        );
        assert_eq!(data_file, *manifest.entries()[0].data_file());
    }
}

#[cfg(test)]
mod merge_tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use tempfile::TempDir;

    use crate::io::LocalFsStorageFactory;
    use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, ManifestStatus,
        NestedField, PrimitiveType, Schema, Struct, Type,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

    async fn new_table(format: FormatVersion, dir: &TempDir) -> (impl Catalog, Table) {
        let warehouse = format!("file://{}", dir.path().join("warehouse").display());
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
            )
            .await
            .unwrap();
        let namespace = NamespaceIdent::new("ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        let table = catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("merge".to_string())
                    .schema(schema)
                    .format_version(format)
                    .build(),
            )
            .await
            .unwrap();
        (catalog, table)
    }

    /// Appends one file of `n + 1` rows, merging manifests from `merge` if given.
    async fn append(
        catalog: &impl Catalog,
        table: &Table,
        n: usize,
        merge: Option<usize>,
    ) -> Table {
        let file = DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::Data)
            .file_path(format!("{}/data/f{n}.parquet", table.metadata().location()))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(n as u64 + 1)
            .partition(Struct::empty())
            .build()
            .unwrap();
        let tx = Transaction::new(table);
        let mut action = tx.fast_append().add_data_files([file]);
        if let Some(min) = merge {
            action = action.with_manifest_merge(min);
        }
        action.apply(tx).unwrap().commit(catalog).await.unwrap()
    }

    /// Every live file of the table's current snapshot: path to (rows, first row id, snapshot id, sequence number).
    async fn live_files(
        table: &Table,
    ) -> (
        BTreeMap<String, (u64, Option<i64>, Option<i64>, Option<i64>)>,
        usize,
    ) {
        let snapshot = table.metadata().current_snapshot().unwrap();
        let list = table
            .object_cache()
            .get_manifest_list(snapshot, &table.metadata_ref())
            .await
            .unwrap();
        let mut files = BTreeMap::new();
        for manifest_file in list.entries() {
            let manifest = table
                .object_cache()
                .get_manifest(manifest_file)
                .await
                .unwrap();
            for entry in manifest.entries() {
                if entry.status() == ManifestStatus::Deleted {
                    continue;
                }
                let file = entry.data_file();
                let seen = files.insert(
                    file.file_path().to_string(),
                    (
                        file.record_count(),
                        file.first_row_id(),
                        entry.snapshot_id(),
                        entry.sequence_number(),
                    ),
                );
                assert!(seen.is_none(), "{} is in two manifests", file.file_path());
            }
        }
        (files, list.entries().len())
    }

    async fn merges_without_changing_the_files(format: FormatVersion) {
        let dir = TempDir::new().unwrap();
        let (catalog, mut table) = new_table(format, &dir).await;
        // What each file had when its own commit made it: rows, first row id, snapshot, sequence number.
        let mut at_commit = BTreeMap::new();
        let mut most_manifests = 0;
        for n in 0..23 {
            table = append(&catalog, &table, n, Some(5)).await;
            let (files, manifests) = live_files(&table).await;
            most_manifests = most_manifests.max(manifests);
            let name = format!("f{n}.parquet");
            let (path, v) = files
                .iter()
                .find(|(path, _)| path.ends_with(&name))
                .unwrap_or_else(|| panic!("{name} is not in the table after its commit"));
            at_commit.insert(path.clone(), *v);
        }
        let (files, manifests) = live_files(&table).await;
        // Without merging there would be 23; with 5 to merge there are never many more than that.
        assert!(manifests < 12, "{manifests} manifests after 23 commits");
        assert!(most_manifests < 12, "{most_manifests} manifests at most");
        assert_eq!(files.len(), 23);
        for (path, now) in &files {
            let then = at_commit[path];
            assert_eq!(then.0, now.0, "{path}: rows");
            assert_eq!(then.1, now.1, "{path}: first row id");
            assert_eq!(then.3, now.3, "{path}: sequence number");
            // The snapshot that added a file stays the one that did, though its manifest was rewritten.
            assert_eq!(then.2, now.2, "{path}: snapshot id");
        }
        // No row id is given twice.
        let mut ranges: Vec<(i64, i64)> = files
            .values()
            .filter_map(|(rows, first, ..)| first.map(|f| (f, f + *rows as i64)))
            .collect();
        ranges.sort();
        for pair in ranges.windows(2) {
            assert!(pair[0].1 <= pair[1].0, "row ids overlap: {pair:?}");
        }
        if format == FormatVersion::V3 {
            assert_eq!(ranges.len(), 23);
        }
    }

    #[tokio::test]
    async fn manifest_merge_keeps_every_file_v2() {
        merges_without_changing_the_files(FormatVersion::V2).await;
    }

    #[tokio::test]
    async fn manifest_merge_keeps_every_file_v3() {
        merges_without_changing_the_files(FormatVersion::V3).await;
    }

    #[tokio::test]
    async fn the_next_table_version_takes_the_parsed_manifests() {
        let dir = TempDir::new().unwrap();
        let (catalog, mut table) = new_table(FormatVersion::V2, &dir).await;
        for n in 0..6 {
            table = append(&catalog, &table, n, None).await;
        }
        // Parse everything into this version's cache.
        let (_, manifests) = live_files(&table).await;
        assert_eq!(manifests, 6);

        let next = append(&catalog, &table, 6, None).await;
        let ident = table.identifier().clone();
        let reloaded = catalog.load_table(&ident).await.unwrap();
        assert_eq!(reloaded.metadata_location(), next.metadata_location());
        let (reloaded, carried) = reloaded.with_manifests_of(&table).await.unwrap();
        // The six manifests the version before had; the seventh is new.
        assert_eq!(carried, 6);
        let (files, manifests) = live_files(&reloaded).await;
        assert_eq!((files.len(), manifests), (7, 7));
    }

    /// What S3 Tables' compaction leaves: for each file it rewrites, one manifest adding the new file and one that
    /// only records the old file as deleted. They merge too, and as in Java the deletions of earlier snapshots are left out.
    #[tokio::test]
    async fn manifest_merge_consolidates_delete_only_manifests_and_drops_old_deletions() {
        use uuid::Uuid;

        use crate::spec::{ManifestWriterBuilder, Struct};
        use crate::transaction::snapshot::{
            ManifestProcess, MergeManifestProcess, SnapshotProducer,
        };

        let dir = TempDir::new().unwrap();
        let (_catalog, table) = new_table(FormatVersion::V2, &dir).await;
        let schema = table.metadata().current_schema().clone();
        let spec = table.metadata().default_partition_spec().as_ref().clone();
        let file = |name: String, rows: u64| {
            DataFileBuilder::default()
                .partition_spec_id(0)
                .content(DataContentType::Data)
                .file_path(format!(
                    "{}/data/{name}.parquet",
                    table.metadata().location()
                ))
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(100)
                .record_count(rows)
                .partition(Struct::empty())
                .build()
                .unwrap()
        };
        let output = |tag: String| {
            table
                .file_io()
                .new_output(format!(
                    "{}/metadata/merge-test-{tag}.avro",
                    table.metadata().location()
                ))
                .unwrap()
        };
        let compaction_snapshot = 99;
        let mut manifests = Vec::new();
        for i in 0..4i64 {
            let mut writer = ManifestWriterBuilder::new(
                output(format!("add{i}")),
                Some(compaction_snapshot),
                schema.clone(),
                spec.clone(),
            )
            .build_v2_data();
            writer
                .add_file(file(format!("new{i}"), 100), 50 + i)
                .unwrap();
            let mut added = writer.write_manifest_file().await.unwrap();
            added.sequence_number = 60;
            added.min_sequence_number = 50;
            manifests.push(added);

            let mut writer = ManifestWriterBuilder::new(
                output(format!("del{i}")),
                Some(compaction_snapshot),
                schema.clone(),
                spec.clone(),
            )
            .build_v2_data();
            writer
                .add_delete_file(file(format!("old{i}"), 10), 3 + i, Some(3 + i))
                .unwrap();
            let mut deleted = writer.write_manifest_file().await.unwrap();
            deleted.sequence_number = 60;
            deleted.min_sequence_number = 3;
            assert!(deleted.has_deleted_files() && !deleted.has_added_files());
            manifests.push(deleted);
        }

        let mut producer = SnapshotProducer::new(&table, Uuid::new_v4(), HashMap::new(), vec![]);
        let process = MergeManifestProcess {
            min_count: 5,
            target_size_bytes: 8 * 1024 * 1024,
        };
        let merged = process
            .process_manifests(&mut producer, manifests)
            .await
            .unwrap();
        assert_eq!(merged.len(), 1, "eight manifests became one");
        assert_eq!(merged[0].existing_files_count, Some(4));
        assert_eq!(merged[0].deleted_files_count, Some(0));

        let manifest = table.object_cache().get_manifest(&merged[0]).await.unwrap();
        let mut seen = BTreeMap::new();
        for entry in manifest.entries() {
            let name = entry
                .data_file()
                .file_path()
                .rsplit('/')
                .next()
                .unwrap()
                .to_string();
            seen.insert(
                name,
                (entry.status(), entry.snapshot_id(), entry.sequence_number()),
            );
        }
        assert_eq!(
            seen.len(),
            4,
            "the four deletions of the earlier snapshot are gone"
        );
        for i in 0..4i64 {
            assert_eq!(
                seen[&format!("new{i}.parquet")],
                (
                    ManifestStatus::Existing,
                    Some(compaction_snapshot),
                    Some(50 + i)
                ),
                "a live file keeps the snapshot and sequence number it had"
            );
            assert!(!seen.contains_key(&format!("old{i}.parquet")));
        }
    }

    /// A pack of manifests that only record deletions of earlier snapshots leaves nothing to write.
    #[tokio::test]
    async fn manifest_merge_of_only_old_deletions_leaves_no_manifest() {
        use uuid::Uuid;

        use crate::spec::{ManifestWriterBuilder, Struct};
        use crate::transaction::snapshot::{
            ManifestProcess, MergeManifestProcess, SnapshotProducer,
        };

        let dir = TempDir::new().unwrap();
        let (_catalog, table) = new_table(FormatVersion::V2, &dir).await;
        let schema = table.metadata().current_schema().clone();
        let spec = table.metadata().default_partition_spec().as_ref().clone();
        let mut manifests = Vec::new();
        for i in 0..6i64 {
            let file = DataFileBuilder::default()
                .partition_spec_id(0)
                .content(DataContentType::Data)
                .file_path(format!(
                    "{}/data/old{i}.parquet",
                    table.metadata().location()
                ))
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(100)
                .record_count(10)
                .partition(Struct::empty())
                .build()
                .unwrap();
            let output = table
                .file_io()
                .new_output(format!(
                    "{}/metadata/only-del-{i}.avro",
                    table.metadata().location()
                ))
                .unwrap();
            let mut writer =
                ManifestWriterBuilder::new(output, Some(99), schema.clone(), spec.clone())
                    .build_v2_data();
            writer.add_delete_file(file, 3 + i, Some(3 + i)).unwrap();
            let mut manifest = writer.write_manifest_file().await.unwrap();
            manifest.sequence_number = 60;
            manifest.min_sequence_number = 3;
            manifests.push(manifest);
        }
        let mut producer = SnapshotProducer::new(&table, Uuid::new_v4(), HashMap::new(), vec![]);
        let merged = MergeManifestProcess {
            min_count: 5,
            target_size_bytes: 8 * 1024 * 1024,
        }
        .process_manifests(&mut producer, manifests)
        .await
        .unwrap();
        assert!(merged.is_empty(), "{} manifests left", merged.len());
    }
}
