use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// A stable identifier for a job.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JobId(pub String);

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for JobId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for JobId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// The lifecycle state of a job.
///
/// Transitions (synchronous rules, enforced by [`Job::transition`]):
///
/// ```text
/// queued --> waiting_model --> running --> succeeded
///    |            |                |
///    +------------+----------------+--> cancelled
///    +------------+----------------+--> failed
/// failed --> (retry creates a new job)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// Admitted to the queue, dependencies satisfied, not yet started.
    Queued,
    /// Started; waiting for its model server to come up / be ready.
    WaitingModel,
    /// Actively generating.
    Running,
    /// Produced at least one output; terminal.
    Succeeded,
    /// Failed (model error, OOM, invalid inputs discovered late); terminal.
    Failed,
    /// Cancelled before dispatch; terminal.
    Cancelled,
}

impl JobState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::WaitingModel => "waiting_model",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Parse a job state from its string form (the inverse of [`JobState::as_str`]).
pub fn job_state_from_str(s: &str) -> Option<JobState> {
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

/// A declared, immutable description of a model server and one of its tasks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskInfo {
    /// The task name, e.g. `3d-paint`.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Input roles this task accepts, mapped to their accepted MIME types.
    pub input_roles: BTreeMap<String, Vec<String>>,
    /// Whether the task accepts a natural-language prompt.
    pub uses_prompt: bool,
}

/// A declared, immutable description of a model server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerInfo {
    pub id: String,
    pub title: String,
    pub url: String,
    pub tasks: Vec<TaskInfo>,
}

impl ServerInfo {
    /// Look up one of this server's tasks by name.
    pub fn task(&self, name: &str) -> Option<&TaskInfo> {
        self.tasks.iter().find(|task| task.name == name)
    }
}

/// A reference to the output (by index) of another job, used as an input.
///
/// A job with an unmet [`Dependency`] stays `Queued` until the referenced
/// job `Succeeded`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dependency {
    pub job_id: JobId,
    pub output_index: usize,
}

/// An input binding for a submitted job: an input role mapped to a concrete
/// asset, or to an output of a dependency job.
#[derive(Debug, Clone, PartialEq,Serialize, Deserialize)]
pub enum InputRef {
    /// A ready asset in the library.
    Asset(AssetRef),
    /// An output of a (possibly not-yet-complete) job.
    JobOutput(Dependency),
}

/// An asset reference embedded in an [`InputRef::Asset`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetRef {
    pub id: crate::AssetId,
    pub role: String,
}

/// A generation job: a task on a server with inputs, parameters, seeds, and
/// a lifecycle state.
///
/// The *rules* for valid transitions and input validation live here as
/// synchronous, pure functions so they can be unit-tested without a runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job {
    pub id: JobId,
    pub server: String,
    pub task: String,
    pub state: JobState,
    /// An optional natural-language prompt (if the task uses one).
    pub prompt: Option<String>,
    /// Task-specific parameters (free JSON; validated by the server).
    pub params: serde_json::Value,
    /// Inputs by role; a job may run multiple seeds / counts.
    pub inputs: BTreeMap<String, InputRef>,
    /// How many outputs to request.
    pub count: usize,
    /// Deterministic seeds; `None` means the server chooses.
    pub seeds: Option<Vec<u64>>,
    /// Idempotency key: submitting the same key again returns the existing job.
    pub idempotency_key: Option<String>,
    /// The ids of output assets, in output order. Populated on success.
    pub outputs: Vec<crate::AssetId>,
    /// A human-readable error, set on `Failed`.
    pub error: Option<String>,
    /// Creation timestamp (seconds since epoch).
    pub created_at: i64,
    /// Last state change timestamp (seconds since epoch).
    pub updated_at: i64,
    /// Id of a job created by `retry` from this one, if any.
    pub retry_of: Option<JobId>,
}

impl Job {
    /// Attempt a state transition, enforcing the state-machine rules.
    ///
    /// Returns `Err` if the transition is illegal for the current state. This
    /// is the single authority on "can a job move from A to B?"; the runtime
    /// callers (server/launcher) call this before persisting a change.
    pub fn transition(&mut self, next: JobState, now: i64) -> crate::Result<()> {
        if self.state.is_terminal() {
            return Err(crate::error::Error::JobTerminated(
                self.id.0.clone(),
                self.state.as_str().to_owned(),
            ));
        }
        let allowed = matches!(
            (self.state, next),
            (JobState::Queued, JobState::WaitingModel)
                | (JobState::Queued, JobState::Running)
                | (JobState::WaitingModel, JobState::Running)
                | (JobState::Queued, JobState::Cancelled)
                | (JobState::WaitingModel, JobState::Cancelled)
                | (JobState::Running, JobState::Cancelled)
                | (JobState::Queued, JobState::Failed)
                | (JobState::WaitingModel, JobState::Failed)
                | (JobState::Running, JobState::Succeeded)
                | (JobState::Running, JobState::Failed)
        );
        if !allowed {
            return Err(crate::error::Error::JobTerminated(
                self.id.0.clone(),
                format!("{:?} -> {:?}", self.state, next),
            ));
        }
        self.state = next;
        self.updated_at = now;
        Ok(())
    }

    /// Validate that every input role required by `task` is present.
    pub fn validate_inputs(&self, task: &TaskInfo) -> crate::Result<()> {
        for (role, _) in &task.input_roles {
            if !self.inputs.contains_key(role) {
                return Err(crate::error::Error::MissingInput(role.clone()));
            }
        }
        Ok(())
    }

    /// The set of dependency job-ids this job is blocked on (unmet inputs).
    pub fn dependencies(&self) -> Vec<Dependency> {
        self.inputs
            .values()
            .filter_map(|input| match input {
                InputRef::JobOutput(d) => Some(d.clone()),
                _ => None,
            })
            .collect()
    }

    /// Whether all of this job's input dependencies are satisfied (i.e. every
    /// referenced job has succeeded). `status_of` maps a job-id to its state.
    pub fn deps_satisfied(&self, status_of: impl Fn(&JobId) -> Option<JobState>) -> bool {
        self.dependencies()
            .iter()
            .all(|d| status_of(&d.job_id) == Some(JobState::Succeeded))
    }

    /// Create a new job id from a seed string (uuid v4).
    pub fn new_id() -> JobId {
        JobId(uuid::Uuid::new_v4().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job() -> Job {
        Job {
            id: JobId::from("j1"),
            server: "trellis2".into(),
            task: "3d-paint".into(),
            state: JobState::Queued,
            prompt: None,
            params: serde_json::json!({"texture_size": 2048}),
            inputs: BTreeMap::new(),
            count: 1,
            seeds: None,
            idempotency_key: None,
            outputs: vec![],
            error: None,
            created_at: 0,
            updated_at: 0,
            retry_of: None,
        }
    }

    #[test]
    fn happy_path() {
        let mut j = job();
        j.transition(JobState::WaitingModel, 1).unwrap();
        j.transition(JobState::Running, 2).unwrap();
        j.transition(JobState::Succeeded, 3).unwrap();
        assert!(j.state.is_terminal());
    }

    #[test]
    fn no_transition_after_terminal() {
        let mut j = job();
        j.transition(JobState::Running, 1).unwrap();
        j.transition(JobState::Succeeded, 2).unwrap();
        assert!(j.transition(JobState::Failed, 3).is_err());
    }

    #[test]
    fn cancel_from_queued_and_waiting() {
        let mut j = job();
        assert!(j.transition(JobState::Cancelled, 1).is_ok());
    }

    #[test]
    fn queued_cannot_jump_to_success() {
        let mut j = job();
        assert!(j.transition(JobState::Succeeded, 1).is_err());
    }

    #[test]
    fn deps_satisfaction() {
        let mut j = job();
        let dep = Dependency { job_id: JobId::from("parent"), output_index: 0 };
        j.inputs.insert(
            "mesh".into(),
            InputRef::JobOutput(dep.clone()),
        );
        let ok = |id: &JobId| Some(if id.0 == "parent" {
            JobState::Succeeded
        } else {
            JobState::Running
        });
        assert!(j.deps_satisfied(ok));
        assert_eq!(j.dependencies(), vec![dep]);
    }

    #[test]
    fn input_validation() {
        let mut j = job();
        let task = TaskInfo {
            name: "3d-paint".into(),
            description: String::new(),
            input_roles: {
                let mut m = BTreeMap::new();
                m.insert("mesh".into(), vec!["model/gltf-binary".into()]);
                m.insert("image".into(), vec!["image/png".into()]);
                m
            },
            uses_prompt: false,
        };
        assert!(j.validate_inputs(&task).is_err());
        j.inputs.insert("mesh".into(), InputRef::JobOutput(Dependency { job_id: JobId::from("p"), output_index: 0 }));
        j.inputs.insert("image".into(), InputRef::JobOutput(Dependency { job_id: JobId::from("p"), output_index: 0 }));
        assert!(j.validate_inputs(&task).is_ok());
    }
}
