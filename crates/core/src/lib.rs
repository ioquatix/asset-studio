//! Scheduler-agnostic domain model and job state machine for asset-studio.
//!
//! This crate defines the data types and the *synchronous* state-machine
//! rules only. It performs no I/O and has no dependency on a scheduler
//! runtime; the orchestration (async work, sockets, processes) lives in the
//! `server`, `launcher`, `library`, and `models` crates. Keeping the domain
//! pure means it can be unit-tested with a plain `std` test harness.

pub mod asset;
pub mod error;
pub mod job;
pub mod job_input;
pub mod time;

pub use asset::{Asset, AssetKind, AssetId};
pub use error::{Error, Result};
pub use job::{Dependency, InputRef, Job, JobId, JobState, ServerInfo, TaskInfo};
pub use job_input::{resolve_inputs, ResolvedInput};

/// Re-exported from `job` for convenience (embedded in [`InputRef`]).
pub use job::AssetRef;
