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

use std::sync::Arc;

use async_trait::async_trait;

use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Result, TableRequirement, TableUpdate};

/// A transaction action that removes the schemas of a table that nothing refers to: every schema but the current one and
/// those its snapshots name (and so its refs, which name snapshots), as Java's metadata cleanup does.
///
/// Table metadata keeps each schema a table ever had, so a table whose columns are added often carries a metadata file that
/// grows with the number of changes times the table's width, rewritten on every commit. A schema a kept snapshot names is
/// kept, since readers of that snapshot (Java's `SnapshotUtil.schemaFor`) look it up; it goes once the snapshot expires.
/// The commit requires the current schema to be the one it was planned against, so a schema added meanwhile is never
/// removed.
pub struct RemoveSchemasAction;

impl RemoveSchemasAction {
    pub(crate) fn new() -> Self {
        RemoveSchemasAction
    }
}

#[async_trait]
impl TransactionAction for RemoveSchemasAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let metadata = table.metadata();
        let current_schema_id = metadata.current_schema_id();
        let named: std::collections::HashSet<i32> =
            metadata.snapshots().filter_map(|s| s.schema_id()).collect();
        let schema_ids: Vec<i32> = metadata
            .schemas_iter()
            .map(|s| s.schema_id())
            .filter(|id| *id != current_schema_id && !named.contains(id))
            .collect();
        if schema_ids.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }
        Ok(ActionCommit::new(
            vec![TableUpdate::RemoveSchemas { schema_ids }],
            vec![TableRequirement::CurrentSchemaIdMatch { current_schema_id }],
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::transaction::Transaction;
    use crate::transaction::action::TransactionAction;
    use crate::transaction::tests::make_v2_table;
    use crate::{TableRequirement, TableUpdate};

    #[tokio::test]
    async fn every_schema_nothing_refers_to_is_removed() {
        let table = make_v2_table();
        let current = table.metadata().current_schema_id();
        assert!(table.metadata().schemas_iter().len() > 1);
        let action = Arc::new(Transaction::new(&table).remove_schemas());
        let mut commit = action.commit(&table).await.unwrap();
        let updates = commit.take_updates();
        let requirements = commit.take_requirements();
        let named: Vec<i32> = table
            .metadata()
            .snapshots()
            .filter_map(|s| s.schema_id())
            .collect();
        let expected: Vec<i32> = table
            .metadata()
            .schemas_iter()
            .map(|s| s.schema_id())
            .filter(|id| *id != current && !named.contains(id))
            .collect();
        assert_eq!(
            updates,
            vec![TableUpdate::RemoveSchemas {
                schema_ids: expected
            }]
        );
        assert_eq!(
            requirements,
            vec![TableRequirement::CurrentSchemaIdMatch {
                current_schema_id: current
            }]
        );
    }
}
