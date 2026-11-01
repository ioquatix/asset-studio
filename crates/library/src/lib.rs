//! SQLite-backed asset library and content-addressed blob store.
//!
//! The [`Library`] is the persistence edge: it stores asset *metadata*, job
//! records, declared servers, and the raw *bytes* of every asset. Bytes live
//! in a content-addressed blob store on disk (addressed by SHA-256, two-level
//! directory sharding), so identical files are stored exactly once and reads
//! are O(1) by digest. Metadata lives in SQLite (assets, jobs, servers tables).
//!
//! All operations are synchronous (`rusqlite` + `std::fs`). Async wrappers are
//! added by the `server`/`launcher` crates via the scheduler's `Blocking`
//! capability.
use std::path::Path;

pub mod blob;
pub mod schema;

pub use blob::BlobStore;
pub use schema::SCHEMA;

use asset_studio_core::{
    Asset, AssetId, AssetKind, InputRef, Job, JobId, JobState, ServerInfo,
};
use rusqlite::{params, Connection};
use std::str::FromStr;

/// Errors from the persistence layer.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("domain error: {0}")]
    Core(#[from] asset_studio_core::Error),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

/// The persistence layer: SQLite metadata + a content-addressed blob store.
pub struct Library {
    conn: Connection,
    blobs: BlobStore,
}

impl Library {
    /// Open (or create) a library rooted at `root`. Creates the schema and the
    /// `blobs/` directory if missing.
    pub fn open(root: &Path) -> Result<Self, StoreError> {
        let blobs = BlobStore::open(root.join("blobs"))?;
        let conn = Connection::open(root.join("library.db"))?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn, blobs })
    }

    /// The content-addressed blob store.
    pub fn blobs(&self) -> &BlobStore {
        &self.blobs
    }

    // ---- assets -----------------------------------------------------------

    /// Insert/overwrite an asset, writing its bytes to the blob store first
    /// (deduped by SHA-256). Returns the stored asset with computed digest and
    /// size.
    pub fn upsert_asset(&self, asset: &Asset, bytes: &[u8]) -> Result<Asset, StoreError> {
        let sha = blob_sha256(bytes);
        self.blobs.write(&sha, bytes)?;
        let mut a = asset.clone();
        a.blob_sha256 = sha;
        a.size_bytes = bytes.len() as u64;
        self.conn.execute(
            "INSERT INTO assets (
                id, kind, mime, title, size_bytes, blob_sha256, rating,
                tags, producer, created_at, deleted_at
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
             ON CONFLICT(id) DO UPDATE SET
                kind=excluded.kind, mime=excluded.mime, title=excluded.title,
                size_bytes=excluded.size_bytes, blob_sha256=excluded.blob_sha256,
                rating=excluded.rating, tags=excluded.tags,
                producer=excluded.producer, created_at=excluded.created_at,
                deleted_at=excluded.deleted_at",
            params![
                a.id.0,
                a.kind.as_str(),
                a.mime,
                a.title,
                a.size_bytes as i64,
                a.blob_sha256,
                a.rating,
                serde_json::to_string(&a.tags).unwrap_or_default(),
                a.producer,
                a.created_at,
                a.deleted_at,
            ],
        )?;
        Ok(a)
    }

    /// Load a live (non-deleted) asset by id.
    pub fn get_asset(&self, id: &AssetId) -> Result<Option<Asset>, StoreError> {
        Ok(self.get_asset_raw(id)?.filter(|a| a.is_live()))
    }

    /// Load an asset regardless of deletion state.
    pub fn get_asset_raw(&self, id: &AssetId) -> Result<Option<Asset>, StoreError> {
        let result = self.conn.query_row(
            "SELECT id, kind, mime, title, size_bytes, blob_sha256, rating, tags, producer,
                    created_at, deleted_at FROM assets WHERE id = ?1",
            params![id.0],
            map_asset,
        );
        Ok(match result {
            Ok(a) => Some(a),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => return Err(e.into()),
        })
    }

    /// Read the bytes of an asset from the blob store.
    pub fn read_asset_bytes(&self, id: &AssetId) -> Result<Vec<u8>, StoreError> {
        let asset = self
            .get_asset_raw(id)?
            .ok_or(asset_studio_core::Error::UnknownAsset(id.0.clone()))?;
        Ok(self.blobs.read(&asset.blob_sha256, asset.size_bytes as usize)?)
    }
    /// List live assets, newest first, optionally filtered by kind.
    pub fn list_assets(&self, kind: Option<&str>, limit: i64) -> Result<Vec<Asset>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, mime, title, size_bytes, blob_sha256, rating, tags, producer,
                    created_at, deleted_at
             FROM assets
             WHERE deleted_at IS NULL AND (?1 IS NULL OR kind = ?1)
             ORDER BY created_at DESC, id
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![kind, limit], map_asset)?;
        let out: Vec<Asset> = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(out)
    }

    /// Soft-delete an asset.
    pub fn delete_asset(&self, id: &AssetId, now: i64) -> Result<(), StoreError> {
        self.conn.execute(
            "UPDATE assets SET deleted_at = ?2 WHERE id = ?1",
            params![id.0, now],
        )?;
        Ok(())
    }

    /// Set (or clear) an asset's rating.
    pub fn set_rating(&self, id: &AssetId, rating: Option<u8>) -> Result<(), StoreError> {
        self.conn
            .execute("UPDATE assets SET rating = ?2 WHERE id = ?1", params![id.0, rating])?;
        Ok(())
    }

    // ---- jobs -------------------------------------------------------------

    /// Insert a new job. Errors with [`asset_studio_core::Error::DuplicateIdempotencyKey`]
    /// if its idempotency key is already taken.
    pub fn insert_job(&self, job: &Job) -> Result<(), StoreError> {
        if let Some(key) = &job.idempotency_key {
            let dup: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM jobs WHERE idempotency_key = ?1",
                params![key],
                |r| r.get(0),
            )?;
            if dup > 0 {
                return Err(StoreError::Core(
                    asset_studio_core::Error::DuplicateIdempotencyKey,
                ));
            }
        }
        self.conn.execute(
            "INSERT INTO jobs (
                id, server, task, state, prompt, params, inputs, count, seeds,
                idempotency_key, outputs, error, created_at, updated_at, retry_of
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
            params![
                job.id.0,
                job.server,
                job.task,
                job.state.as_str(),
                job.prompt,
                job.params.to_string(),
                serde_json::to_string(&job.inputs).map_err(sql_err)?,
                job.count as i64,
                encode_seeds(job.seeds.as_deref()),
                job.idempotency_key,
                serde_json::to_string(&job.outputs).map_err(sql_err)?,
                job.error,
                job.created_at,
                job.updated_at,
                job.retry_of.as_ref().map(|j| j.0.clone()),
            ],
        )?;
        Ok(())
    }

    /// Load a job by id.
    pub fn get_job(&self, id: &JobId) -> Result<Option<Job>, StoreError> {
        let result = self
            .conn
            .query_row("SELECT * FROM jobs WHERE id = ?1", params![id.0], map_job);
        Ok(match result {
            Ok(j) => Some(j),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => return Err(e.into()),
        })
    }

    /// List jobs, newest first, optionally filtered by state.
    pub fn list_jobs(&self, state: Option<&str>, limit: i64) -> Result<Vec<Job>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM jobs WHERE (?1 IS NULL OR state = ?1) ORDER BY created_at DESC, id LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![state, limit], map_job)?;
        let out: Vec<Job> = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(out)
    }

    /// Persist a job's current state (transitions, outputs, error).
    pub fn update_job(&self, job: &Job) -> Result<(), StoreError> {
        self.conn.execute(
            "UPDATE jobs SET state=?2, prompt=?3, params=?4, inputs=?5, count=?6,
                seeds=?7, outputs=?8, error=?9, updated_at=?10, retry_of=?11 WHERE id=?1",
            params![
                job.id.0,
                job.state.as_str(),
                job.prompt,
                job.params.to_string(),
                serde_json::to_string(&job.inputs).map_err(sql_err)?,
                job.count as i64,
                encode_seeds(job.seeds.as_deref()),
                serde_json::to_string(&job.outputs).map_err(sql_err)?,
                job.error,
                job.updated_at,
                job.retry_of.as_ref().map(|j| j.0.clone()),
            ],
        )?;
        Ok(())
    }

    /// All job ids and their states (for dependency checks).
    pub fn job_states(&self) -> Result<Vec<(JobId, JobState)>, StoreError> {
        let mut stmt = self.conn.prepare("SELECT id, state FROM jobs")?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let state: String = row.get(1)?;
            Ok((JobId(id), job_state_from_str(&state).unwrap_or(JobState::Queued)))
        })?;
        let out: Vec<(JobId, JobState)> = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(out)
    }

    /// The output asset-ids of a job, in order.
    pub fn job_outputs(&self, id: &JobId) -> Result<Vec<AssetId>, StoreError> {
        Ok(self
            .get_job(id)?
            .ok_or(asset_studio_core::Error::UnknownAsset(id.0.clone()))?
            .outputs)
    }

    // ---- servers ----------------------------------------------------------

    /// Upsert a declared server.
    pub fn upsert_server(&self, server: &ServerInfo) -> Result<(), StoreError> {
        let tasks = serde_json::to_string(&server.tasks).unwrap_or_default();
        self.conn.execute(
            "INSERT INTO servers (id, title, url, tasks) VALUES (?1,?2,?3,?4)
             ON CONFLICT(id) DO UPDATE SET title=excluded.title, url=excluded.url, tasks=excluded.tasks",
            params![server.id, server.title, server.url, tasks],
        )?;
        Ok(())
    }

    /// All declared servers.
    pub fn list_servers(&self) -> Result<Vec<ServerInfo>, StoreError> {
        let mut stmt = self.conn.prepare("SELECT id, title, url, tasks FROM servers")?;
        let rows = stmt.query_map([], |row| {
            let tasks: String = row.get("tasks")?;
            Ok(ServerInfo {
                id: row.get("id")?,
                title: row.get("title")?,
                url: row.get("url")?,
                tasks: serde_json::from_str(&tasks).unwrap_or_default(),
            })
        })?;
        let out: Vec<ServerInfo> = rows.collect::<Result<Vec<_>, _>>()?;
        Ok(out)
    }
}

// ---- helpers --------------------------------------------------------------

fn blob_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

fn sql_err(e: serde_json::Error) -> rusqlite::Error {
    rusqlite::Error::InvalidParameterName(e.to_string())
}

fn encode_seeds(seeds: Option<&[u64]>) -> Option<String> {
    seeds.map(|s| serde_json::to_string(s).unwrap_or_default())
}

fn decode_seeds(s: &Option<String>) -> Option<Vec<u64>> {
    s.as_deref().and_then(|x| serde_json::from_str(x).ok())
}

/// Parse a job state from its string form (inverse of `as_str`).
fn job_state_from_str(s: &str) -> Option<JobState> {
    match s {
        "queued" => Some(JobState::Queued),
        "waiting_model" => Some(JobState::WaitingModel),
        "running" => Some(JobState::Running),
        "succeeded" => Some(JobState::Succeeded),
        "failed" => Some(JobState::Failed),
        "cancelled" => Some(JobState::Cancelled),
        _ => None,
    }
}

fn map_asset(row: &rusqlite::Row<'_>) -> rusqlite::Result<Asset> {
    let kind: String = row.get("kind")?;
    Ok(Asset {
        id: AssetId(row.get("id")?),
        kind: AssetKind::from_str(&kind).unwrap_or(AssetKind::Image),
        mime: row.get("mime")?,
        title: row.get("title")?,
        size_bytes: row.get::<_, i64>("size_bytes")? as u64,
        blob_sha256: row.get("blob_sha256")?,
        rating: row.get("rating")?,
        tags: serde_json::from_str(&row.get::<_, String>("tags")?).unwrap_or_default(),
        producer: row.get("producer")?,
        created_at: row.get("created_at")?,
        deleted_at: row.get("deleted_at")?,
    })
}

fn map_job(row: &rusqlite::Row<'_>) -> rusqlite::Result<Job> {
    let state: String = row.get("state")?;
    let inputs: String = row.get("inputs")?;
    let outputs: String = row.get("outputs")?;
    let params: String = row.get("params")?;
    let seeds: Option<String> = row.get("seeds")?;
    Ok(Job {
        id: JobId(row.get("id")?),
        server: row.get("server")?,
        task: row.get("task")?,
        state: job_state_from_str(&state).unwrap_or(JobState::Queued),
        prompt: row.get("prompt")?,
        params: serde_json::from_str(&params).unwrap_or(serde_json::Value::Null),
        inputs: deserialize_inputs(&inputs),
        count: row.get::<_, i64>("count")? as usize,
        seeds: decode_seeds(&seeds),
        idempotency_key: row.get("idempotency_key")?,
        outputs: deserialize_outputs(&outputs),
        error: row.get("error")?,
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
        retry_of: row.get::<_, Option<String>>("retry_of")?.map(JobId),
    })
}

fn deserialize_inputs(s: &str) -> std::collections::BTreeMap<String, InputRef> {
    serde_json::from_str(s).unwrap_or_default()
}

fn deserialize_outputs(s: &str) -> Vec<AssetId> {
    serde_json::from_str(s).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use asset_studio_core::AssetKind;

    fn test_asset(id: &str) -> Asset {
        Asset {
            id: AssetId::from(id),
            kind: AssetKind::Image,
            mime: "image/png".into(),
            title: Some("test".into()),
            size_bytes: 0,
            blob_sha256: String::new(),
            rating: None,
            tags: vec!["style:test".into()],
            producer: None,
            created_at: 100,
            deleted_at: None,
        }
    }

    fn open_lib() -> (tempfile::TempDir, Library) {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::open(dir.path()).unwrap();
        (dir, lib)
    }

    #[test]
    fn asset_roundtrip_and_dedup() {
        let (_d, lib) = open_lib();
        let bytes = b"png-bytes";
        let stored = lib.upsert_asset(&test_asset("a1"), bytes).unwrap();
        assert_eq!(stored.size_bytes, bytes.len() as u64);
        assert!(!stored.blob_sha256.is_empty());

        let got = lib.read_asset_bytes(&AssetId::from("a1")).unwrap();
        assert_eq!(got, bytes);

        // A second asset with identical bytes shares the blob.
        let a2 = lib.upsert_asset(&test_asset("a2"), bytes).unwrap();
        assert_eq!(a2.blob_sha256, stored.blob_sha256);
        assert!(lib.blobs().contains(&stored.blob_sha256));
    }

    #[test]
    fn list_and_soft_delete() {
        let (_d, lib) = open_lib();
        lib.upsert_asset(&test_asset("a1"), b"x").unwrap();
        lib.upsert_asset(&test_asset("a2"), b"y").unwrap();
        assert_eq!(lib.list_assets(None, 10).unwrap().len(), 2);

        lib.delete_asset(&AssetId::from("a1"), 200).unwrap();
        let live = lib.list_assets(None, 10).unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].id.0, "a2");
        // Raw access still sees the deleted row.
        assert!(lib.get_asset_raw(&AssetId::from("a1")).unwrap().is_some());
        // Live access does not.
        assert!(lib.get_asset(&AssetId::from("a1")).unwrap().is_none());
    }

    #[test]
    fn rating_and_tags() {
        let (_d, lib) = open_lib();
        lib.upsert_asset(&test_asset("a1"), b"x").unwrap();
        lib.set_rating(&AssetId::from("a1"), Some(4)).unwrap();
        let a = lib.get_asset(&AssetId::from("a1")).unwrap().unwrap();
        assert_eq!(a.rating, Some(4));
        assert_eq!(a.tags, vec!["style:test".to_string()]);
    }

    fn test_job(id: &str, state: JobState) -> Job {
        Job {
            id: JobId::from(id),
            server: "trellis2".into(),
            task: "3d-paint".into(),
            state,
            prompt: Some("a red chicken".into()),
            params: serde_json::json!({"texture_size": 2048}),
            inputs: std::collections::BTreeMap::new(),
            count: 1,
            seeds: Some(vec![1, 2, 3]),
            idempotency_key: None,
            outputs: vec![],
            error: None,
            created_at: 1,
            updated_at: 1,
            retry_of: None,
        }
    }

    #[test]
    fn job_roundtrip_and_idempotency() {
        let (_d, lib) = open_lib();
        let mut job = test_job("j1", JobState::Queued);
        job.idempotency_key = Some("key-1".into());
        lib.insert_job(&job).unwrap();

        // Duplicate idempotency key is rejected.
        assert!(lib.insert_job(&job).is_err());

        // Transition to running, then succeeded with an output.
        let mut loaded = lib.get_job(&JobId::from("j1")).unwrap().unwrap();
        loaded.transition(JobState::Running, 2).unwrap();
        loaded.transition(JobState::Succeeded, 3).unwrap();
        loaded.outputs = vec![AssetId::from("out1")];
        lib.update_job(&loaded).unwrap();

        let after = lib.get_job(&JobId::from("j1")).unwrap().unwrap();
        assert_eq!(after.state, JobState::Succeeded);
        assert_eq!(after.outputs, vec![AssetId::from("out1")]);
        assert_eq!(after.seeds, Some(vec![1, 2, 3]));
        assert!(after.params.get("texture_size").is_some());
    }

    #[test]
    fn list_jobs_by_state() {
        let (_d, lib) = open_lib();
        lib.insert_job(&test_job("j1", JobState::Queued)).unwrap();
        lib.insert_job(&test_job("j2", JobState::Running)).unwrap();
        assert_eq!(lib.list_jobs(Some("queued"), 10).unwrap().len(), 1);
        assert_eq!(lib.list_jobs(None, 10).unwrap().len(), 2);
    }

    #[test]
    fn servers_roundtrip() {
        let (_d, lib) = open_lib();
        let server = ServerInfo {
            id: "trellis2".into(),
            title: "TRELLIS.2 Paint".into(),
            url: "http://127.0.0.1:8123".into(),
            tasks: vec![asset_studio_core::TaskInfo {
                name: "3d-paint".into(),
                description: "paint a mesh".into(),
                input_roles: serde_json::from_str(
                    r#"{"mesh":["model/gltf-binary"],"image":["image/png"]}"#,
                )
                .unwrap(),
                uses_prompt: false,
            }],
        };
        lib.upsert_server(&server).unwrap();
        let servers = lib.list_servers().unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].tasks[0].name, "3d-paint");
    }
}
