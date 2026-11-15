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

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::{MAIN_BRANCH, SnapshotReference, SnapshotRetention};
use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, Result, TableRequirement, TableUpdate};

/// A transaction action that creates and removes tags, after Java's `ManageSnapshots`.
///
/// Every change carries a requirement the catalog checks against its current table at commit,
/// not against the table the transaction was built from:
/// - [`create_tag`](Self::create_tag) requires that no ref of that name exists;
/// - [`remove_tag`](Self::remove_tag) requires that the tag still points where it pointed when the
///   action was committed;
/// - [`assert_ref`](Self::assert_ref) adds a requirement alone: a ref points at a snapshot, or is
///   absent.
///
/// A requirement that names a fixed snapshot ([`create_tag`](Self::create_tag) and
/// [`assert_ref`](Self::assert_ref)) cannot pass on a retry once it fails, so a failure against
/// the table a commit attempt reloads is returned as a non-retryable
/// [`ErrorKind::CatalogCommitConflicts`]. The catalog's own check closes the window between that
/// reload and the commit.
///
/// Combines with other actions in one transaction (a property update and a tag land together).
/// Each ref may be named once per action.
pub struct ManageSnapshotsAction {
    changes: Vec<RefChange>,
}

enum RefChange {
    CreateTag {
        name: String,
        snapshot_id: i64,
    },
    RemoveTag {
        name: String,
    },
    AssertRef {
        name: String,
        snapshot_id: Option<i64>,
    },
}

impl RefChange {
    fn name(&self) -> &str {
        match self {
            RefChange::CreateTag { name, .. }
            | RefChange::RemoveTag { name }
            | RefChange::AssertRef { name, .. } => name,
        }
    }
}

impl ManageSnapshotsAction {
    pub(crate) fn new() -> Self {
        Self { changes: vec![] }
    }

    /// Tags `snapshot_id` as `name`, a tag with no age limit of its own (it then lives as long as
    /// `history.expire.max-ref-age-ms`, unbounded by default). The commit fails if a branch or tag
    /// named `name` exists in the catalog's table at commit, or the snapshot is not in it.
    pub fn create_tag(mut self, name: impl Into<String>, snapshot_id: i64) -> Self {
        self.changes.push(RefChange::CreateTag {
            name: name.into(),
            snapshot_id,
        });
        self
    }

    /// Removes the tag `name`. The commit fails if there is no such tag, or it is a branch.
    pub fn remove_tag(mut self, name: impl Into<String>) -> Self {
        self.changes
            .push(RefChange::RemoveTag { name: name.into() });
        self
    }

    /// Requires that the branch or tag `name` points at `snapshot_id` at commit, or is absent for
    /// `None`; changes nothing. For example, `assert_ref("main", Some(s))` alongside
    /// `create_tag("t", s)` tags `s` only while it is still the current snapshot.
    pub fn assert_ref(mut self, name: impl Into<String>, snapshot_id: Option<i64>) -> Self {
        self.changes.push(RefChange::AssertRef {
            name: name.into(),
            snapshot_id,
        });
        self
    }

    /// A failed fixed requirement, which no retry can pass.
    fn check_fixed(requirement: &TableRequirement, table: &Table) -> Result<()> {
        requirement
            .check(Some(table.metadata()))
            .map_err(|e| e.with_retryable(false))
    }
}

#[async_trait]
impl TransactionAction for ManageSnapshotsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let metadata = table.metadata();

        let mut seen = HashSet::new();
        if let Some(change) = self.changes.iter().find(|c| !seen.insert(c.name())) {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                format!(
                    "Ref `{}` is named more than once in one action",
                    change.name()
                ),
            ));
        }

        let mut updates = vec![];
        let mut requirements = vec![];
        for change in &self.changes {
            match change {
                RefChange::CreateTag { name, snapshot_id } => {
                    if name == MAIN_BRANCH {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            "Cannot create a tag named main: main must be a branch",
                        ));
                    }
                    let requirement = TableRequirement::RefSnapshotIdMatch {
                        r#ref: name.clone(),
                        snapshot_id: None,
                    };
                    Self::check_fixed(&requirement, table)?;
                    if metadata.snapshot_by_id(*snapshot_id).is_none() {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Cannot tag `{name}`: snapshot {snapshot_id} is not in the table"
                            ),
                        ));
                    }
                    requirements.push(requirement);
                    updates.push(TableUpdate::SetSnapshotRef {
                        ref_name: name.clone(),
                        reference: SnapshotReference::new(*snapshot_id, SnapshotRetention::Tag {
                            max_ref_age_ms: None,
                        }),
                    });
                }
                RefChange::RemoveTag { name } => {
                    let Some(reference) = metadata.refs().get(name) else {
                        return Err(Error::new(
                            ErrorKind::CatalogCommitConflicts,
                            format!("Cannot remove tag `{name}`: it does not exist"),
                        ));
                    };
                    if reference.is_branch() {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!("Cannot remove `{name}` as a tag: it is a branch"),
                        ));
                    }
                    requirements.push(TableRequirement::RefSnapshotIdMatch {
                        r#ref: name.clone(),
                        snapshot_id: Some(reference.snapshot_id),
                    });
                    updates.push(TableUpdate::RemoveSnapshotRef {
                        ref_name: name.clone(),
                    });
                }
                RefChange::AssertRef { name, snapshot_id } => {
                    let requirement = TableRequirement::RefSnapshotIdMatch {
                        r#ref: name.clone(),
                        snapshot_id: *snapshot_id,
                    };
                    Self::check_fixed(&requirement, table)?;
                    requirements.push(requirement);
                }
            }
        }

        Ok(ActionCommit::new(updates, requirements))
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, MAIN_BRANCH, SnapshotRetention,
        Struct,
    };
    use crate::table::Table;
    use crate::transaction::action::{ApplyTransactionAction, TransactionAction};
    use crate::transaction::tests::make_v3_minimal_table_in_catalog;
    use crate::transaction::{ManageSnapshotsAction, Transaction};
    use crate::{Catalog, ErrorKind, TableCommit};

    async fn append(catalog: &impl Catalog, table: &Table, seq: u32) -> Table {
        let file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("test/{seq}.parquet"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .build()
            .unwrap();
        let tx = Transaction::new(table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![file])
            .apply(tx)
            .unwrap();
        tx.commit(catalog).await.unwrap()
    }

    fn current(table: &Table) -> i64 {
        table.metadata().current_snapshot_id().unwrap()
    }

    async fn tag(
        catalog: &impl Catalog,
        table: &Table,
        name: &str,
        id: i64,
    ) -> crate::Result<Table> {
        let tx = Transaction::new(table);
        let tx = tx.manage_snapshots().create_tag(name, id).apply(tx)?;
        tx.commit(catalog).await
    }

    #[tokio::test]
    async fn test_create_and_remove_tag() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append(&catalog, &table, 1).await;
        let first = current(&table);
        let table = append(&catalog, &table, 2).await;

        let table = tag(&catalog, &table, "cut", first).await.unwrap();
        let reference = table.metadata().refs().get("cut").unwrap();
        assert_eq!(reference.snapshot_id, first);
        assert_eq!(reference.retention, SnapshotRetention::Tag {
            max_ref_age_ms: None
        });
        // main is untouched.
        assert_ne!(current(&table), first);
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(
            reloaded.metadata().refs().get("cut").unwrap().snapshot_id,
            first
        );

        let tx = Transaction::new(&reloaded);
        let tx = tx.manage_snapshots().remove_tag("cut").apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert!(!table.metadata().refs().contains_key("cut"));
        assert!(table.metadata().refs().contains_key(MAIN_BRANCH));
    }

    #[tokio::test]
    async fn test_refused_changes() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append(&catalog, &table, 1).await;
        let first = current(&table);
        let table = tag(&catalog, &table, "cut", first).await.unwrap();

        // A tag on a name in use, a branch included.
        for name in ["cut", MAIN_BRANCH] {
            let err = tag(&catalog, &table, name, first).await.unwrap_err();
            assert!(!err.retryable(), "{err}");
        }
        let err = tag(&catalog, &table, "other", 42).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid);

        // Removing a missing tag, or a branch.
        for name in ["missing", MAIN_BRANCH] {
            let tx = Transaction::new(&table);
            let tx = tx.manage_snapshots().remove_tag(name).apply(tx).unwrap();
            assert!(tx.commit(&catalog).await.is_err(), "{name}");
        }

        // One name twice in one action.
        let tx = Transaction::new(&table);
        let err = tx
            .manage_snapshots()
            .create_tag("a", first)
            .assert_ref("a", None)
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::PreconditionFailed);

        let table = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(table.metadata().refs().len(), 2);
    }

    /// A transaction built before an append landed is checked against the table at commit: its
    /// `main` requirement fails and nothing is written.
    #[tokio::test]
    async fn test_stale_main_requirement_fails_commit() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let loaded = append(&catalog, &table, 1).await;
        let s = current(&loaded);

        let tx = Transaction::new(&loaded);
        let tx = tx
            .manage_snapshots()
            .assert_ref(MAIN_BRANCH, Some(s))
            .create_tag("cut", s)
            .apply(tx)
            .unwrap();

        // Another writer appends between load and commit.
        append(&catalog, &loaded, 2).await;

        let err = tx.commit(&catalog).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
        assert!(!err.retryable());
        let table = catalog.load_table(loaded.identifier()).await.unwrap();
        assert!(!table.metadata().refs().contains_key("cut"));
    }

    /// The catalog checks the requirements against its own current table, so an append that lands
    /// after the action was planned (between a commit attempt's reload and its catalog update)
    /// fails the commit.
    #[tokio::test]
    async fn test_catalog_checks_requirements_against_current_table() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let loaded = append(&catalog, &table, 1).await;
        let s = current(&loaded);

        let planned = |table: &Table| {
            let action = ManageSnapshotsAction::new()
                .assert_ref(MAIN_BRANCH, Some(s))
                .create_tag("cut", s);
            let table = table.clone();
            async move {
                let mut commit = std::sync::Arc::new(action).commit(&table).await.unwrap();
                TableCommit::builder()
                    .ident(table.identifier().clone())
                    .updates(commit.take_updates())
                    .requirements(commit.take_requirements())
                    .build()
            }
        };

        let commit = planned(&loaded).await;
        append(&catalog, &loaded, 2).await;
        let err = catalog.update_table(commit).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);

        // Planned and committed against the same table, it lands.
        let fresh = catalog.load_table(loaded.identifier()).await.unwrap();
        let tx = Transaction::new(&fresh);
        let tx = tx
            .manage_snapshots()
            .assert_ref(MAIN_BRANCH, Some(current(&fresh)))
            .create_tag("cut", s)
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(table.metadata().refs().get("cut").unwrap().snapshot_id, s);

        // A second tag of the same name, planned before the first landed, fails in the catalog.
        let commit = {
            let action = ManageSnapshotsAction::new().create_tag("again", s);
            let mut commit = std::sync::Arc::new(action).commit(&fresh).await.unwrap();
            TableCommit::builder()
                .ident(fresh.identifier().clone())
                .updates(commit.take_updates())
                .requirements(commit.take_requirements())
                .build()
        };
        tag(&catalog, &table, "again", s).await.unwrap();
        let err = catalog.update_table(commit).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    }

    #[tokio::test]
    async fn test_expiry_keeps_tagged_snapshot_until_tag_removed() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append(&catalog, &table, 1).await;
        let tagged = current(&table);
        let table = tag(&catalog, &table, "cut", tagged).await.unwrap();
        let mut table = table;
        for seq in 2..6 {
            table = append(&catalog, &table, seq).await;
        }

        let expire = |table: &Table| {
            let tx = Transaction::new(table);
            tx.expire_snapshots()
                .expire_older_than_ms(i64::MAX)
                .retain_last(1)
                .apply(tx)
                .unwrap()
        };

        let table = expire(&table).commit(&catalog).await.unwrap();
        let ids: Vec<i64> = table
            .metadata()
            .snapshots()
            .map(|s| s.snapshot_id())
            .collect();
        assert_eq!(ids.len(), 2, "{ids:?}");
        assert!(ids.contains(&tagged));
        assert!(ids.contains(&current(&table)));

        let tx = Transaction::new(&table);
        let tx = tx.manage_snapshots().remove_tag("cut").apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        // Removing the tag alone keeps the snapshot; the next expiry drops it.
        assert!(table.metadata().snapshot_by_id(tagged).is_some());
        let table = expire(&table).commit(&catalog).await.unwrap();
        assert!(table.metadata().snapshot_by_id(tagged).is_none());
        assert_eq!(table.metadata().snapshots().count(), 1);
    }

    #[tokio::test]
    async fn test_property_and_tag_in_one_commit() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;
        let table = append(&catalog, &table, 1).await;
        let s = current(&table);
        let metadata_location = table.metadata_location().unwrap().to_string();

        let tx = Transaction::new(&table);
        let tx = tx
            .update_table_properties()
            .set("generation".to_string(), "2".to_string())
            .apply(tx)
            .unwrap();
        let tx = tx
            .manage_snapshots()
            .assert_ref(MAIN_BRANCH, Some(s))
            .create_tag("generation-2", s)
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        assert_eq!(
            table.metadata().properties().get("generation"),
            Some(&"2".to_string())
        );
        assert_eq!(
            table
                .metadata()
                .refs()
                .get("generation-2")
                .unwrap()
                .snapshot_id,
            s
        );
        // One commit after the append's.
        assert_eq!(
            table
                .metadata()
                .metadata_log()
                .last()
                .unwrap()
                .metadata_file,
            metadata_location
        );
    }
}
