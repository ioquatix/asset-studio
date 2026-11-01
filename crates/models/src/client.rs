//! A typed HTTP client for asset-studio model servers.
//!
//! [`ModelClient`] wraps a [`base_http_client::Client`] scoped to a model
//! server's endpoint and exposes the operations the studio performs:
//! `health` (readiness probe), `submit` (start a generation), `job` (poll a
//! job), `jobs` (list jobs). Each call builds a `base_http::Request`, awaits
//! the response, reads the body to completion, and deserializes JSON.
use base_http::body::{BodyFrame, BufferedBody, ReadableBody};
use base_http::{Method, Request, Response, Status};
use base_http_client::{Client, Driver, Endpoint, Options, ResponseBody};
use base_scheduler::Cancellation;
use std::net::SocketAddr;

/// Errors from a model-server call.
#[derive(Debug, thiserror::Error)]
pub enum CallError {
    #[error("transport error: {0}")]
    Transport(#[from] base_http_client::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("http {0}: {1}")]
    Http(u16, String),
}

/// A single HTTP exchange result: final status + full body bytes.
pub struct HttpResult {
    pub status: Status,
    pub body: Vec<u8>,
}

/// A typed client bound to one model server.
///
/// The client borrows the scheduler for `'scope` (it drives the pooled
/// connections on that scheduler). The paired [`Driver`] returned by
/// [`connect`](Self::connect) must be driven concurrently — typically by
/// scheduling it with [`Spawn`](base_scheduler::Spawn) — so calls progress.
pub struct ModelClient<'scope> {
    /// The base address of the model server (e.g. `127.0.0.1:8123`).
    pub address: SocketAddr,
    client: Client<'scope>,
}

impl<'scope> ModelClient<'scope> {
    /// Connect to `address` on `scheduler`, returning a client and its driver.
    pub fn connect<S>(
        scheduler: &'scope S,
        address: SocketAddr,
    ) -> Result<(Self, Driver<'scope>), base_http_client::Error>
    where
        S: base_scheduler::SocketOpen
            + base_scheduler::SocketConnect
            + base_scheduler::SocketClose
            + base_scheduler::SocketSetOption<base_scheduler::TcpNoDelay>
            + base_scheduler::SocketStreamReading
            + base_scheduler::SocketStreamWriting
            + 'scope,
        S::Socket: 'static,
    {
        let endpoint = Endpoint::new(address);
        let (client, driver) = Client::new(scheduler, endpoint, Options::default())?;
        Ok((
            Self {
                address,
                client,
            },
            driver,
        ))
    }

    /// Readiness probe: `GET /health`. Returns the final HTTP status.
    pub async fn health(&self, cancellation: &Cancellation) -> Result<Status, CallError> {
        Ok(self
            .request(Method::GET, "/health", None, cancellation)
            .await?
            .status)
    }

    /// Submit a generation: `POST /generate` with a JSON body.
    pub async fn submit(
        &self,
        payload: &serde_json::Value,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        let body = serde_json::to_vec(payload)?;
        let result = self
            .request(Method::POST, "/generate", Some(&body), cancellation)
            .await?;
        Self::ensure_ok(&result)?;
        Ok(serde_json::from_slice(&result.body)?)
    }

    /// Poll a job: `GET /jobs/{id}`.
    pub async fn job(
        &self,
        id: &str,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        let result = self
            .request(Method::GET, &format!("/jobs/{id}"), None, cancellation)
            .await?;
        Self::ensure_ok(&result)?;
        Ok(serde_json::from_slice(&result.body)?)
    }

    /// List jobs: `GET /jobs`.
    pub async fn jobs(&self, cancellation: &Cancellation) -> Result<serde_json::Value, CallError> {
        let result = self.request(Method::GET, "/jobs", None, cancellation).await?;
        Self::ensure_ok(&result)?;
        Ok(serde_json::from_slice(&result.body)?)
    }

    /// One request + read-to-end.
    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&[u8]>,
        cancellation: &Cancellation,
    ) -> Result<HttpResult, CallError> {
        let request = match body {
            Some(bytes) => Request::new(method, path, BufferedBody::from(bytes.to_vec())),
            None => Request::new(method, path, BufferedBody::from(Vec::new())),
        };
        let response: Response<ResponseBody> = self.client.call(request, cancellation).await?;
        let status = response.metadata.status;
        let body = read_to_end(response.body, cancellation).await?;
        Ok(HttpResult { status, body })
    }

    fn ensure_ok(result: &HttpResult) -> Result<(), CallError> {
        if result.status.is_success() {
            Ok(())
        } else {
            Err(CallError::Http(
                result.status.as_u16(),
                String::from_utf8_lossy(&result.body).to_string(),
            ))
        }
    }
}

/// Read a `ResponseBody` to completion, returning all payload bytes.
async fn read_to_end(
    mut body: ResponseBody,
    cancellation: &Cancellation,
) -> Result<Vec<u8>, base_http_client::Error> {
    let mut out = Vec::new();
    loop {
        match body.read(cancellation).await? {
            BodyFrame::Data { data, .. } => out.extend_from_slice(data.as_ref()),
            BodyFrame::Headers { .. } => {}
            BodyFrame::End => break,
        }
    }
    Ok(out)
}
