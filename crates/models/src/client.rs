//! A typed HTTP client for asset-studio model servers.
//!
//! The model servers speak contract v1 (see `model_server_sdk/contract.py`):
//!
//! * `GET /v1/info` — readiness + model/task info; `status` is
//!   `loading | ready | busy | error`.
//! * `POST /v1/generate` — *synchronous*; it blocks until the model finishes
//!   and returns the outputs inline as base64 (`{role, mime, data_b64}`
//!   inputs → `{mime, data_b64, meta}` outputs). There is **no** job polling:
//!   a request is a job.
//!
//! [`ModelClient`] wraps a [`base_http_client::Client`] scoped to a model
//! server's endpoint and exposes `info` (readiness) and `generate`
//! (one generation). Each call builds a `base_http::Request`, awaits the
//! response, reads the body, and decodes. Errors carry the model's
//! [`ModelError`] detail (code/message/retryable) so the orchestrator can
//! distinguish `busy`, `not_ready`, and `generation_failed`.
use base_http::body::{BodyFrame, BufferedBody, ReadableBody};
use base_http::{Method, Request, Response, Status};
use base_http_client::{Client, Driver, Endpoint, Options, ResponseBody};
use base_scheduler::Cancellation;
use std::net::SocketAddr;

/// A model-server contract error, as returned by a non-2xx response
/// (`{"error": {code, message, retryable}}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError {
    /// The HTTP status (e.g. 409, 503, 500).
    pub status: u16,
    /// The contract error code: `busy` | `not_ready` | `unsupported_task` |
    /// `invalid_request` | `generation_failed` | `internal`.
    pub code: String,
    pub message: String,
    /// Whether the orchestrator should retry (e.g. `busy`, `not_ready`).
    pub retryable: bool,
}

/// Errors from a model-server call.
#[derive(Debug, thiserror::Error)]
pub enum CallError {
    #[error("transport error: {0}")]
    Transport(#[from] base_http_client::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// A well-formed success body that failed to decode (e.g. a missing
    /// `data_b64` field).
    #[error("decode error: {0}")]
    Decoding(String),
    /// The server answered with a non-2xx status (contract error).
    #[error("model error [{code}]: {message}")]
    Model {
        code: String,
        message: String,
        status: u16,
        retryable: bool,
    },
}

impl CallError {
    /// The contract [`ModelError`], if this was a model-level error.
    pub fn model(&self) -> Option<ModelError> {
        match self {
            CallError::Model {
                code,
                message,
                status,
                retryable,
            } => Some(ModelError {
                status: *status,
                code: code.clone(),
                message: message.clone(),
                retryable: *retryable,
            }),
            _ => None,
        }
    }

    /// True when the orchestrator should retry the operation.
    pub fn retryable(&self) -> bool {
        match self {
            CallError::Model { retryable, .. } => *retryable,
            // Transport errors (connection reset/dropped) are retryable too.
            CallError::Transport(_) => true,
            _ => false,
        }
    }
}

/// One decode-ready output from `generate`.
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub mime: String,
    /// The decoded payload bytes.
    pub data: Vec<u8>,
    /// Contract meta (e.g. `triangles`, `surface_error_relative`, `seed`).
    pub meta: serde_json::Value,
}

/// The result of a `generate` call.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub seed: u64,
    /// Effective parameters (request values with defaults filled in).
    pub params: serde_json::Value,
    pub outputs: Vec<GenerateOutput>,
    pub generate_s: f64,
}

/// The readiness of a model server from `GET /v1/info`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerStatusInfo {
    /// `loading` | `ready` | `busy` | `error`.
    pub status: String,
    pub status_message: Option<String>,
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
/// scheduling it with [`Spawn`](base_scheduler::Spawn) or zipping it with the
/// generation — so calls progress.
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

    /// Readiness: `GET /v1/info`. Returns the server's `status` string and an
    /// optional `status_message` (set when the model failed to load).
    pub async fn info(&self, cancellation: &Cancellation) -> Result<ServerStatusInfo, CallError> {
        let result = self.request(Method::GET, "/v1/info", None, cancellation).await?;
        let value: serde_json::Value = serde_json::from_slice(&result.body)?;
        Ok(ServerStatusInfo {
            status: value
                .get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("error")
                .to_string(),
            status_message: value
                .get("status_message")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string()),
        })
    }

    /// `POST /v1/generate` — run one generation synchronously.
    ///
    /// `inputs` are the task's input files as `(role, mime, bytes)`; `params`
    /// is the task's flat params object (empty `{}` for all defaults);
    /// `seed` is optional (the server derives per-output seeds from it);
    /// `count` is how many outputs to request.
    pub async fn generate(
        &self,
        task: &str,
        prompt: Option<&str>,
        inputs: &[(String, String, Vec<u8>)],
        params: &serde_json::Value,
        count: u64,
        seed: Option<u64>,
        cancellation: &Cancellation,
    ) -> Result<GenerateResult, CallError> {
        let inputs_json: Vec<serde_json::Value> = inputs
            .iter()
            .map(|(role, mime, data)| {
                serde_json::json!({
                    "role": role,
                    "mime": mime,
                    "data_b64": data_encoding::BASE64.encode(data),
                })
            })
            .collect();
        let payload = serde_json::json!({
            "task": task,
            "prompt": prompt,
            "inputs": inputs_json,
            "params": params,
            "count": count,
            "seed": seed,
        });
        let body = serde_json::to_vec(&payload)?;
        let result = self
            .request(Method::POST, "/v1/generate", Some(&body), cancellation)
            .await?;
        Self::ensure_ok(&result)?;
        Ok(Self::decode_generate(&result.body)?)
    }

    /// Decode a `GenerateResponse` JSON body into a [`GenerateResult`],
    /// base64-decoding each output.
    fn decode_generate(body: &[u8]) -> Result<GenerateResult, CallError> {
        let value: serde_json::Value = serde_json::from_slice(body)?;
        let mut outputs = Vec::new();
        if let Some(list) = value.get("outputs").and_then(|o| o.as_array()) {
            for out in list {
                let mime = out
                    .get("mime")
                    .and_then(|m| m.as_str())
                    .unwrap_or_default()
                    .to_string();
                let data_b64 = out
                    .get("data_b64")
                    .and_then(|d| d.as_str())
                    .ok_or_else(|| CallError::Decoding("missing data_b64".into()))?;
                let data = data_encoding::BASE64
                    .decode(data_b64.as_bytes())
                    .map_err(|e| CallError::Decoding(e.to_string()))?;
                let meta = out.get("meta").cloned().unwrap_or(serde_json::Value::Null);
                outputs.push(GenerateOutput { mime, data, meta });
            }
        }
        Ok(GenerateResult {
            seed: value.get("seed").and_then(|s| s.as_u64()).unwrap_or(0),
            params: value.get("params").cloned().unwrap_or(serde_json::Value::Null),
            outputs,
            generate_s: value
                .pointer("/timing/generate_s")
                .and_then(|t| t.as_f64())
                .unwrap_or(0.0),
        })
    }

    fn ensure_ok(result: &HttpResult) -> Result<(), CallError> {
        if result.status.is_success() {
            return Ok(());
        }
        // Parse the contract error detail if present; fall back to the raw body.
        let detail: serde_json::Value =
            serde_json::from_slice(&result.body).unwrap_or(serde_json::Value::Null);
        let err = detail.get("error");
        match (err.and_then(|e| e.get("code")), err.and_then(|e| e.get("message")), err.and_then(|e| e.get("retryable"))) {
            (Some(code), Some(message), retryable) if code.is_string() => {
                Err(CallError::Model {
                    code: code.as_str().unwrap_or_default().to_string(),
                    message: message.as_str().unwrap_or_default().to_string(),
                    status: result.status.as_u16(),
                    retryable: retryable.and_then(|r| r.as_bool()).unwrap_or(false),
                })
            }
            _ => Err(CallError::Model {
                code: "http".into(),
                message: String::from_utf8_lossy(&result.body).to_string(),
                status: result.status.as_u16(),
                retryable: false,
            }),
        }
    }

    /// One request + read-to-end.
    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&[u8]>,
        cancellation: &Cancellation,
    ) -> Result<HttpResult, CallError> {
        let request = model_request(method, path, body);
        let response: Response<ResponseBody> = self.client.call(request, cancellation).await?;
        let status = response.metadata.status;
        let body = read_to_end(response.body, cancellation).await?;
        Ok(HttpResult { status, body })
    }
}

/// Model endpoints exchange JSON; without an explicit media type FastAPI
/// can interpret a generation body as bytes instead of a request object.
fn model_request(method: Method, path: &str, body: Option<&[u8]>) -> Request<BufferedBody<Vec<u8>>> {
    let mut request = Request::new(
        method,
        path,
        BufferedBody::from(body.unwrap_or_default().to_vec()),
    );
    request.fields
        .add("Accept", "application/json")
        .expect("static HTTP field");
    if body.is_some() {
        request.fields
            .add("Content-Type", "application/json")
            .expect("static HTTP field");
    }
    request
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_requests_declare_json_content_type() {
        let request = model_request(Method::POST, "/v1/generate", Some(b"{}"));
        assert_eq!(
            request.fields.values("content-type").collect::<Vec<_>>(),
            vec![b"application/json".as_slice()],
        );
        assert_eq!(
            request.fields.values("accept").collect::<Vec<_>>(),
            vec![b"application/json".as_slice()],
        );
        let info = model_request(Method::GET, "/v1/info", None);
        assert_eq!(info.fields.values("content-type").count(), 0);
    }

    #[test]
    fn decodes_a_generate_response() {
        let glb = b"fake-glb-bytes";
        let b64 = data_encoding::BASE64.encode(glb);
        let body = serde_json::json!({
            "model": {"id": "trellis2-paint"},
            "task": "3d-paint",
            "seed": 42,
            "params": {"texture_size": 2048},
            "outputs": [{"mime": "model/gltf-binary", "data_b64": b64, "meta": {"triangles": 100}}],
            "timing": {"generate_s": 1.5},
        })
        .to_string();
        let result = ModelClient::decode_generate(body.as_bytes()).unwrap();
        assert_eq!(result.seed, 42);
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].data, glb);
        assert_eq!(result.outputs[0].meta["triangles"], 100);
        assert_eq!(result.generate_s, 1.5);
    }

    #[test]
    fn parses_model_error_detail() {
        let body = serde_json::json!({ "error": { "code": "busy", "message": "another generation is running", "retryable": true } }).to_string();
        let result = HttpResult {
            status: Status::CONFLICT,
            body: body.into_bytes(),
        };
        match ModelClient::ensure_ok(&result) {
            Err(CallError::Model { code, retryable, .. }) => {
                assert_eq!(code, "busy");
                assert!(retryable);
            }
            other => panic!("expected Model error, got {other:?}"),
        }
    }
}
