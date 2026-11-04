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
use futures::future::try_join_all;
use uuid::Uuid;

use crate::error::{Result, invalid_data};
use crate::spec::{DataFile, ManifestContentType, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{ManifestProcess, SnapshotProduceOperation, SnapshotProducer};
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind};

/// RewriteFilesAction replaces data files of the table with others in one snapshot (operation
/// `replace`), as a compaction or a merge of the rows of a partition does. It is modelled on Java's
/// `RewriteFiles`.
///
/// Files to delete are named by their path, which is what a caller has from a scan plan
/// (`FileScanTask::data_file_path`). The action finds them in the manifests of the table it is
/// applied to, so the caller need not keep the [`DataFile`] it read.
///
/// # Validation
///
/// Every time the action is applied (on the table the transaction started from, and again on the
/// refreshed table each time a commit conflict makes [`Transaction::commit`](crate::transaction::Transaction::commit)
/// retry, see `Transaction::do_commit`), against that table:
///
/// - every file to delete must be live (`Added` or `Existing`) in the current snapshot's manifests.
///   If one is not (a concurrent rewrite removed it) the commit fails with [`ErrorKind::DataInvalid`]
///   and a message beginning `Cannot delete file that is not live in the table`, naming the path.
///   This error is not retryable: the caller must plan the rewrite again.
/// - an added file must not already be live in the table (`ErrorKind::DataInvalid`).
/// - deleting no file and adding no file is an error (`ErrorKind::PreconditionFailed`); deleting
///   files and adding none is allowed.
/// - a table whose current snapshot has delete manifests (position or equality deletes), or a
///   manifest to rewrite that is not of the default partition spec, is refused with
///   `ErrorKind::FeatureUnsupported`.
///
/// Files that other commits appended since the caller planned the rewrite are kept: only the named
/// files are removed.
///
/// # Manifests
///
/// A manifest that holds a file to delete is rewritten: the deleted entries as `Deleted` with the
/// snapshot's id, the others as `Existing` with the snapshot id, sequence numbers and first row id they
/// had (entries deleted by earlier snapshots are left out). Other manifests are carried unchanged.
/// The added files go into a new manifest as `Added`; on a format version 3 table they are assigned
/// row ids as a fast append's are.
pub struct RewriteFilesAction {
    delete_paths: Vec<String>,
    added_data_files: Vec<DataFile>,
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
}

impl RewriteFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            delete_paths: vec![],
            added_data_files: vec![],
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
        }
    }

    /// Delete the data files at these paths (the `file_path` of their [`DataFile`], the
    /// `data_file_path` of the `FileScanTask` that read them).
    pub fn delete_files<S: Into<String>>(mut self, paths: impl IntoIterator<Item = S>) -> Self {
        self.delete_paths.extend(paths.into_iter().map(Into::into));
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
}

/// What the current snapshot's manifests say about the rewrite.
struct Plan {
    /// The files to delete, as the manifests hold them.
    deleted: Vec<DataFile>,
    /// Paths of the manifests that hold at least one of them.
    affected_manifests: HashSet<String>,
}

impl RewriteFilesAction {
    async fn plan(
        &self,
        table: &Table,
        delete: &HashSet<&str>,
        added: &HashSet<&str>,
    ) -> Result<Plan> {
        let mut plan = Plan {
            deleted: vec![],
            affected_manifests: HashSet::new(),
        };
        let Some(snapshot) = table.metadata().current_snapshot() else {
            return Err(Self::not_live(delete.iter().next().copied()));
        };
        let manifest_list = table.manifest_list_reader(snapshot).load().await?;
        if let Some(m) = manifest_list
            .entries()
            .iter()
            .find(|m| m.content == ManifestContentType::Deletes)
        {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "Cannot rewrite files of a table with delete manifests, found {}",
                    m.manifest_path
                ),
            ));
        }

        let cache = table.object_cache();
        let manifests = try_join_all(manifest_list.entries().iter().map(|m| {
            let cache = cache.clone();
            async move { cache.get_manifest(m).await.map(|manifest| (m, manifest)) }
        }))
        .await?;

        let mut found: HashSet<&str> = HashSet::new();
        for (manifest_file, manifest) in &manifests {
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let path = entry.file_path();
                if delete.contains(path) {
                    if !found.insert(path) {
                        return Err(invalid_data!(
                            "File {path} is live in more than one manifest entry"
                        ));
                    }
                    plan.deleted.push(entry.data_file().clone());
                    plan.affected_manifests
                        .insert(manifest_file.manifest_path.clone());
                } else if added.contains(path) {
                    return Err(invalid_data!(
                        "Cannot add file that is already referenced by table: {path}"
                    ));
                }
            }
        }
        if let Some(missing) = delete.iter().filter(|p| !found.contains(**p)).min() {
            return Err(Self::not_live(Some(missing)));
        }
        for m in manifest_list
            .entries()
            .iter()
            .filter(|m| plan.affected_manifests.contains(&m.manifest_path))
        {
            if m.partition_spec_id != table.metadata().default_partition_spec_id() {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!(
                        "Cannot rewrite manifest {} of partition spec {}, the default is {}",
                        m.manifest_path,
                        m.partition_spec_id,
                        table.metadata().default_partition_spec_id()
                    ),
                ));
            }
        }
        Ok(plan)
    }

    fn not_live(path: Option<&str>) -> Error {
        invalid_data!(
            "Cannot delete file that is not live in the table: {}",
            path.unwrap_or("(no snapshot)")
        )
    }
}

#[async_trait]
impl TransactionAction for RewriteFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // Collapse repeats, so a manifest never names a file twice.
        let delete: HashSet<&str> = self.delete_paths.iter().map(String::as_str).collect();
        let mut seen = HashSet::new();
        let added_files: Vec<DataFile> = self
            .added_data_files
            .iter()
            .filter(|f| seen.insert(f.file_path.as_str()))
            .cloned()
            .collect();
        if delete.is_empty() && added_files.is_empty() {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "A rewrite needs files to delete or files to add",
            ));
        }
        let added: HashSet<&str> = added_files.iter().map(|f| f.file_path.as_str()).collect();

        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.snapshot_properties.clone(),
            added_files.clone(),
        );
        snapshot_producer.validate_added_data_files()?;

        // Against the table as it is now, which is the refreshed one on a retry.
        let plan = self.plan(table, &delete, &added).await?;

        snapshot_producer
            .with_removed_data_files(plan.deleted)
            .commit(RewriteFilesOperation, RewriteManifestsProcess {
                delete_paths: delete.iter().map(|p| p.to_string()).collect(),
                affected_manifests: plan.affected_manifests,
            })
            .await
    }
}

struct RewriteFilesOperation;

impl SnapshotProduceOperation for RewriteFilesOperation {
    fn operation(&self) -> Operation {
        Operation::Replace
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
        // Delete-only manifests are kept: they record removals until `expire_snapshots`.
        Ok(manifest_list
            .entries()
            .iter()
            .filter(|m| m.has_added_files() || m.has_existing_files() || m.has_deleted_files())
            .cloned()
            .collect())
    }
}

/// Rewrites the manifests that hold a file to delete.
struct RewriteManifestsProcess {
    delete_paths: HashSet<String>,
    affected_manifests: HashSet<String>,
}

impl ManifestProcess for RewriteManifestsProcess {
    async fn process_manifests(
        &self,
        snapshot_produce: &mut SnapshotProducer<'_>,
        manifests: Vec<ManifestFile>,
    ) -> Result<Vec<ManifestFile>> {
        let cache = snapshot_produce.table.object_cache();
        let mut out = Vec::with_capacity(manifests.len());
        for manifest_file in manifests {
            if !self
                .affected_manifests
                .contains(&manifest_file.manifest_path)
            {
                out.push(manifest_file);
                continue;
            }
            let manifest = cache.get_manifest(&manifest_file).await?;
            let mut writer = snapshot_produce.new_manifest_writer(ManifestContentType::Data)?;
            let mut kept = 0usize;
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    // Deleted by an earlier snapshot: only a reader of that snapshot needs it.
                    continue;
                }
                let entry = entry.as_ref().clone();
                if entry.sequence_number().is_none() || entry.file_sequence_number.is_none() {
                    return Err(invalid_data!(
                        "Manifest entry of {} has no sequence number, cannot rewrite {}",
                        entry.file_path(),
                        manifest_file.manifest_path
                    ));
                }
                if self.delete_paths.contains(entry.file_path()) {
                    writer.add_delete_entry(entry)?;
                } else {
                    writer.add_existing_entry(entry)?;
                }
                kept += 1;
            }
            if kept > 0 {
                out.push(writer.write_manifest_file().await?);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use futures::TryStreamExt;

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, ManifestStatus,
        Operation, Struct,
    };
    use crate::table::Table;
    use crate::transaction::tests::make_v3_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, ErrorKind};

    fn file(path: &str, rows: u64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(rows * 10)
            .record_count(rows)
            .partition(Struct::from_iter([Some(Literal::long(1))]))
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    async fn append(catalog: &impl Catalog, table: &Table, f: DataFile) -> Table {
        let tx = Transaction::new(table);
        let tx = tx.fast_append().add_data_files([f]).apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    async fn live_paths(table: &Table) -> BTreeSet<String> {
        let tasks: Vec<_> = table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        tasks
            .into_iter()
            .map(|t| t.data_file_path().to_string())
            .collect()
    }

    fn set(paths: &[&str]) -> BTreeSet<String> {
        paths.iter().map(|p| p.to_string()).collect()
    }

    async fn table_with_three() -> (impl Catalog, Table) {
        let catalog = new_memory_catalog().await;
        let mut table = make_v3_minimal_table_in_catalog(&catalog).await;
        for (p, rows) in [("a.parquet", 10), ("b.parquet", 20), ("c.parquet", 30)] {
            table = append(&catalog, &table, file(p, rows)).await;
        }
        (catalog, table)
    }

    #[tokio::test]
    async fn test_rewrite_two_of_three() {
        let (catalog, table) = table_with_three().await;
        let tx = Transaction::new(&table);
        let tx = tx
            .rewrite_files()
            .delete_files(["a.parquet", "c.parquet"])
            .add_data_files([file("new.parquet", 40)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(live_paths(&table).await, set(&["b.parquet", "new.parquet"]));
        let snapshot = table.metadata().current_snapshot().unwrap();
        assert_eq!(snapshot.summary().operation, Operation::Replace);
        let props = &snapshot.summary().additional_properties;
        assert_eq!(props["added-data-files"], "1");
        assert_eq!(props["deleted-data-files"], "2");
        assert_eq!(props["added-records"], "40");
        assert_eq!(props["deleted-records"], "40");
        assert_eq!(props["added-files-size"], "400");
        assert_eq!(props["removed-files-size"], "400");
        assert_eq!(props["total-data-files"], "2");
        assert_eq!(props["total-records"], "60");
        assert_eq!(props["total-files-size"], "600");

        // The deleted entries are marked Deleted with the new snapshot's id.
        let list = table.manifest_list_reader(snapshot).load().await.unwrap();
        let mut deleted = BTreeSet::new();
        for m in list.entries() {
            let manifest = table.object_cache().get_manifest(m).await.unwrap();
            for e in manifest.entries() {
                if e.status == ManifestStatus::Deleted {
                    assert_eq!(e.snapshot_id, Some(snapshot.snapshot_id()));
                    deleted.insert(e.file_path().to_string());
                }
            }
        }
        assert_eq!(deleted, set(&["a.parquet", "c.parquet"]));
        // b.parquet's manifest is carried unchanged; a's and c's are rewritten; the new file has one.
        assert_eq!(list.entries().len(), 4);
    }

    #[tokio::test]
    async fn test_rewrite_keeps_concurrent_append() {
        let (catalog, table) = table_with_three().await;
        // Built against version N...
        let tx = Transaction::new(&table);
        let tx = tx
            .rewrite_files()
            .delete_files(["a.parquet", "b.parquet"])
            .add_data_files([file("new.parquet", 30)])
            .apply(tx)
            .unwrap();
        // ...another writer commits first.
        append(&catalog, &table, file("d.parquet", 5)).await;
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(
            live_paths(&table).await,
            set(&["c.parquet", "d.parquet", "new.parquet"])
        );
    }

    #[tokio::test]
    async fn test_rewrite_of_file_not_live_fails() {
        let (catalog, table) = table_with_three().await;
        let tx = Transaction::new(&table);
        let tx = tx
            .rewrite_files()
            .delete_files(["a.parquet"])
            .add_data_files([file("new1.parquet", 10)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let before = table.metadata().current_snapshot_id();

        let tx = Transaction::new(&table);
        let tx = tx
            .rewrite_files()
            .delete_files(["a.parquet"])
            .add_data_files([file("new2.parquet", 10)])
            .apply(tx)
            .unwrap();
        let err = tx.commit(&catalog).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(!err.retryable());
        assert!(
            err.message()
                .contains("Cannot delete file that is not live in the table: a.parquet"),
            "{err}"
        );
        let table = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(table.metadata().current_snapshot_id(), before);
        assert_eq!(
            live_paths(&table).await,
            set(&["b.parquet", "c.parquet", "new1.parquet"])
        );
    }

    #[tokio::test]
    async fn test_rewrite_that_is_concurrently_invalidated_fails_on_retry() {
        let (catalog, table) = table_with_three().await;
        let tx = Transaction::new(&table);
        let tx = tx
            .rewrite_files()
            .delete_files(["a.parquet"])
            .add_data_files([file("mine.parquet", 10)])
            .apply(tx)
            .unwrap();
        // A concurrent rewrite removes the file first.
        let other = Transaction::new(&table);
        let other = other
            .rewrite_files()
            .delete_files(["a.parquet"])
            .add_data_files([file("theirs.parquet", 10)])
            .apply(other)
            .unwrap();
        let table = other.commit(&catalog).await.unwrap();
        let err = tx.commit(&catalog).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert!(
            live_paths(&table).await.contains("theirs.parquet")
                && !live_paths(&table).await.contains("mine.parquet")
        );
    }

    #[tokio::test]
    async fn test_rewrite_needs_something_to_do_and_can_delete_only() {
        let (catalog, table) = table_with_three().await;
        let tx = Transaction::new(&table);
        let tx = tx.rewrite_files().apply(tx).unwrap();
        let err = tx.commit(&catalog).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::PreconditionFailed);

        let tx = Transaction::new(&table);
        let tx = tx
            .rewrite_files()
            .delete_files(["a.parquet", "b.parquet", "c.parquet"])
            .set_snapshot_properties(HashMap::from([("k".to_string(), "v".to_string())]))
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert!(live_paths(&table).await.is_empty());
        let props = &table
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties;
        assert_eq!(props["total-data-files"], "0");
        assert_eq!(props["total-records"], "0");
        assert_eq!(props["k"], "v");
    }

    #[tokio::test]
    async fn test_rewrite_then_appends_with_manifest_merge() {
        let (catalog, mut table) = table_with_three().await;
        let tx = Transaction::new(&table);
        let tx = tx
            .rewrite_files()
            .delete_files(["a.parquet", "b.parquet"])
            .add_data_files([file("new.parquet", 30)])
            .apply(tx)
            .unwrap();
        table = tx.commit(&catalog).await.unwrap();

        let mut expected = set(&["c.parquet", "new.parquet"]);
        for i in 0..6 {
            let name = format!("later-{i}.parquet");
            let tx = Transaction::new(&table);
            let tx = tx
                .fast_append()
                .with_manifest_merge(2)
                .add_data_files([file(&name, 1)])
                .apply(tx)
                .unwrap();
            table = tx.commit(&catalog).await.unwrap();
            expected.insert(name);
            assert_eq!(live_paths(&table).await, expected);
        }
        // The merge dropped the entries deleted by the rewrite: no manifest names them.
        let snapshot = table.metadata().current_snapshot().unwrap();
        let list = table.manifest_list_reader(snapshot).load().await.unwrap();
        assert!(list.entries().len() < 9, "{}", list.entries().len());
        for m in list.entries() {
            let manifest = table.object_cache().get_manifest(m).await.unwrap();
            for e in manifest.entries() {
                assert_ne!(e.status, ManifestStatus::Deleted, "{}", e.file_path());
            }
        }
        // And a rewrite after the merge still finds the files that were carried through it.
        let tx = Transaction::new(&table);
        let tx = tx
            .rewrite_files()
            .delete_files(["c.parquet", "later-0.parquet"])
            .apply(tx)
            .unwrap();
        table = tx.commit(&catalog).await.unwrap();
        assert!(!live_paths(&table).await.contains("c.parquet"));
    }

    #[tokio::test]
    async fn test_rewrite_keeps_row_lineage_consistent() {
        let (catalog, table) = table_with_three().await;
        assert_eq!(table.metadata().next_row_id(), 60);
        let tx = Transaction::new(&table);
        let tx = tx
            .rewrite_files()
            .delete_files(["a.parquet", "b.parquet"])
            .add_data_files([file("new.parquet", 30)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snapshot = table.metadata().current_snapshot().unwrap();
        assert_eq!(snapshot.first_row_id(), Some(60));
        // Rewritten manifests take row ids for their live rows (as the manifest merge does); the
        // table's next row id is where the snapshot's range ends.
        let assigned = table.metadata().next_row_id() - 60;
        assert!(assigned >= 30, "{assigned}");

        let list = table.manifest_list_reader(snapshot).load().await.unwrap();
        let mut first_ids = HashMap::new();
        for m in list.entries() {
            let manifest = table.object_cache().get_manifest(m).await.unwrap();
            for e in manifest.entries().iter().filter(|e| e.is_alive()) {
                first_ids.insert(e.file_path().to_string(), e.data_file().first_row_id);
            }
        }
        // c.parquet keeps the ids it was given when appended (30..60); the new file's come from the range.
        assert_eq!(first_ids["c.parquet"], Some(30));
        let new_first = first_ids["new.parquet"].unwrap();
        assert!(new_first >= 60 && new_first + 30 <= table.metadata().next_row_id() as i64);
    }
}
