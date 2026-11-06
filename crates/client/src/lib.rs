//! A typed HTTP client for the `asset-studio-server` REST API.
//!
//! [`StudioClient`] mirrors [`ModelClient`](asset_studio_models::ModelClient):
//! it wraps a [`base_http_client::Client`] scoped to the studio endpoint and
//! exposes the operations the CLI performs (list, import, submit, poll,
//! download, …). Each call builds a [`base_http::Request`], awaits the
//! response, reads the body to completion, and deserializes JSON.
//!
//! The client borrows the scheduler for `'scope`; the paired
//! [`Driver`](base_http_client::Driver) returned by [`connect`](Self::connect)
//! must be driven concurrently (zip it with your command) so calls progress.

use base_http::body::{BodyFrame, BufferedBody, ReadableBody};
use base_http::{Method, Request, Response, Status};
use base_http_client::{Client, Driver, Endpoint, Options, ResponseBody};
use base_scheduler::Cancellation;
use std::net::SocketAddr;

/// Errors from a studio API call.
#[derive(Debug, thiserror::Error)]
pub enum CallError {
    #[error("transport error: {0}")]
    Transport(#[from] base_http_client::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("http {0}: {1}")]
    Http(u16, String),
}

/// A single HTTP exchange result: final status + full body bytes.
pub struct HttpResult {
    pub status: Status,
    pub body: Vec<u8>,
}

/// A typed client bound to the studio server.
///
/// The client borrows the scheduler for `'scope` (it drives the pooled
/// connections on that scheduler). The paired [`Driver`] returned by
/// [`connect`](Self::connect) must be driven concurrently — typically by
/// zipping it with the command's future — so calls progress.
pub struct StudioClient<'scope> {
    /// The base address of the studio server (e.g. `127.0.0.1:8090`).
    pub address: SocketAddr,
    client: Client<'scope>,
}

impl<'scope> StudioClient<'scope> {
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

    /// `GET /` — service root (name, server list, status).
    pub async fn root(
        &self,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        Ok(self.get("/", cancellation).await?)
    }

    /// `GET /servers` — registered model servers + their tasks.
    pub async fn servers(
        &self,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        Ok(self.get("/servers", cancellation).await?)
    }

    /// `GET /assets[?kind=]` — list assets.
    pub async fn assets(
        &self,
        kind: Option<&str>,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        let path = match kind {
            Some(k) => format!("/assets?kind={k}"),
            None => "/assets".to_string(),
        };
        Ok(self.get(&path, cancellation).await?)
    }

    /// `GET /assets/{id}` — fetch one asset's metadata.
    pub async fn asset(
        &self,
        id: &str,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        Ok(self.get(&format!("/assets/{id}"), cancellation).await?)
    }

    /// `GET /assets/{id}/download` — fetch an asset's raw bytes.
    pub async fn download(
        &self,
        id: &str,
        cancellation: &Cancellation,
    ) -> Result<Vec<u8>, CallError> {
        let result = self
            .request(Method::GET, &format!("/assets/{id}/download"), None, cancellation)
            .await?;
        if !result.status.is_success() {
            return Err(CallError::Http(
                result.status.as_u16(),
                String::from_utf8_lossy(&result.body).to_string(),
            ));
        }
        Ok(result.body)
    }

    /// `POST /assets/import?kind=&title=` — import raw bytes as an asset.
    pub async fn import(
        &self,
        kind: &str,
        title: Option<&str>,
        bytes: &[u8],
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        let mut path = format!("/assets/import?kind={kind}");
        if let Some(t) = title {
            path.push_str(&format!("&title={t}"));
        }
        self.request(Method::POST, &path, Some(bytes), cancellation)
            .await?
            .parsed()
    }

    /// `GET /jobs[?state=]` — list jobs.
    pub async fn jobs(
        &self,
        state: Option<&str>,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        let path = match state {
            Some(s) => format!("/jobs?state={s}"),
            None => "/jobs".to_string(),
        };
        Ok(self.get(&path, cancellation).await?)
    }

    /// `GET /jobs/{id}` — fetch one job.
    pub async fn job(
        &self,
        id: &str,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        Ok(self.get(&format!("/jobs/{id}"), cancellation).await?)
    }

    /// `POST /jobs` — submit a job from a JSON payload.
    pub async fn submit(
        &self,
        payload: &serde_json::Value,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        let body = serde_json::to_vec(payload)?;
        self.request(Method::POST, "/jobs", Some(&body), cancellation)
            .await?
            .parsed()
    }

    /// `POST /pipelines/text-to-3d` — submit the prompt → picture → cut-out →
    /// GLB chain; returns `{jobs, model_job}`.
    pub async fn text_to_3d(
        &self,
        payload: &serde_json::Value,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        let body = serde_json::to_vec(payload)?;
        self.request(Method::POST, "/pipelines/text-to-3d", Some(&body), cancellation)
            .await?
            .parsed()
    }

    /// `POST /jobs/{id}/cancel` — cancel a job.
    pub async fn cancel(
        &self,
        id: &str,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        self.request(Method::POST, &format!("/jobs/{id}/cancel"), None, cancellation)
            .await?
            .parsed()
    }

    /// `POST /jobs/{id}/retry` — retry a failed job.
    pub async fn retry(
        &self,
        id: &str,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        self.request(Method::POST, &format!("/jobs/{id}/retry"), None, cancellation)
            .await?
            .parsed()
    }

    /// `GET /health` — readiness probe.
    pub async fn health(&self, cancellation: &Cancellation) -> Result<Status, CallError> {
        let result = self.request(Method::GET, "/health", None, cancellation).await?;
        Ok(result.status)
    }

    /// One `GET` request, parsed to JSON.
    async fn get(
        &self,
        path: &str,
        cancellation: &Cancellation,
    ) -> Result<serde_json::Value, CallError> {
        self.request(Method::GET, path, None, cancellation)
            .await?
            .parsed()
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
}

impl HttpResult {
    /// Confirm a success status, then parse the body as JSON.
    pub fn parsed(&self) -> Result<serde_json::Value, CallError> {
        if !self.status.is_success() {
            return Err(CallError::Http(
                self.status.as_u16(),
                String::from_utf8_lossy(&self.body).to_string(),
            ));
        }
        Ok(serde_json::from_slice(&self.body)?)
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
