//! The SQLite schema for the asset-studio library.

/// The full schema. Executed with `execute_batch` on open; `CREATE TABLE IF
/// NOT EXISTS` makes it idempotent across restarts.
pub const SCHEMA: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS assets (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    mime        TEXT NOT NULL,
    title       TEXT,
    size_bytes  INTEGER NOT NULL,
    blob_sha256 TEXT NOT NULL,
    rating      INTEGER,
    tags        TEXT NOT NULL DEFAULT '[]',
    producer    TEXT,
    created_at  INTEGER NOT NULL,
    deleted_at  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(kind);
CREATE INDEX IF NOT EXISTS idx_assets_created ON assets(created_at);

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    server          TEXT NOT NULL,
    task            TEXT NOT NULL,
    state           TEXT NOT NULL,
    prompt          TEXT,
    params          TEXT NOT NULL DEFAULT '{}',
    inputs          TEXT NOT NULL DEFAULT '{}',
    count           INTEGER NOT NULL DEFAULT 1,
    seeds           INTEGER,
    idempotency_key TEXT UNIQUE,
    outputs         TEXT NOT NULL DEFAULT '[]',
    error           TEXT,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    retry_of        TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS servers (
    id    TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url   TEXT NOT NULL,
    tasks TEXT NOT NULL DEFAULT '[]'
);
"#;
