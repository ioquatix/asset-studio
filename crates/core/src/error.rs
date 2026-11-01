use thiserror::Error;

/// A strongly-typed error for the asset-studio domain.
///
/// Transport/scheduler I/O errors are handled in the orchestration crates;
/// these are the *semantic* failures the domain can produce.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("unknown server `{0}`")]
    UnknownServer(String),

    #[error("server `{server}` does not declare task `{task}`")]
    UnknownTask { server: String, task: String },

    #[error("duplicate idempotency key: a job with this key already exists")]
    DuplicateIdempotencyKey,

    #[error("unknown asset `{0}`")]
    UnknownAsset(String),

    #[error("asset `{0}` is deleted")]
    DeletedAsset(String),

    #[error("invalid input for task `{task}`: {detail}")]
    InvalidInput { task: String, detail: String },

    #[error("asset `{asset}` is not an acceptable input for role `{role}`")]
    UnacceptableInput { asset: String, role: String },

    #[error("the job `{0}` is already {1}")]
    JobTerminated(String, String),

    #[error("a job can only be cancelled while queued")]
    CannotCancel,

    #[error("a job can only be retried after it failed")]
    CannotRetry,

    #[error("missing required input role `{0}`")]
    MissingInput(String),

    #[error("invalid asset kind `{0}`")]
    InvalidAssetKind(String),

    #[error("invalid rating: must be 0..=5")]
    InvalidRating,

    #[error("invalid tag `{0}`")]
    InvalidTag(String),

    #[error("the requested generation ran out of admission capacity")]
    OutOfCapacity,
}

/// A `Result` alias using the domain [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
