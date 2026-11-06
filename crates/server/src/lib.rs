//! The asset-studio HTTP API (built on `base-http-server` + `base-scheduler`).
//!
//! [`Studio`] implements the [`Handler`](base_http_server::Handler) trait and
//! serves the REST API over the scheduler's TCP listener. The state machine
//! rules live in [`asset-studio-core`](../asset_studio_core), persistence in
//! [`asset-studio-library`](../asset_studio_library), model-server process
//! management in [`asset-studio-launcher`](../asset_studio_launcher), and the
//! model catalog/HTTP client in [`asset-studio-models`](../asset_studio_models).
//!
//! Routing errors (bad id, unknown task, invalid input, …) are returned as
//! 4xx JSON responses. Only genuine internal failures (store I/O, unexpected
//! panics) become a [`ServerHandlerError`] and let `base-http-server` close the
//! connection.
//!
//! See [`orchestrator`] for the async job pump that dispatches ready jobs to
//! their model servers and reaps running ones.
pub mod orchestrator;

use std::sync::{Arc, Mutex};

use base_http::body::{BodyFrame, BufferedBody, ReadableBody};
use base_http::{InterimHandler, InterimResponses, Request, Response, Status};
use base_http_server::Handler;
use base_scheduler::Cancellation;

use asset_studio_core::JobState;
use asset_studio_library::Library;
use asset_studio_launcher::Launcher;

/// A routing-level failure with a well-defined HTTP status and client message.
/// Returned by the route handlers and converted to a JSON response by
/// [`Studio::route`].
#[derive(Debug)]
enum ApiError {
    /// A domain error (maps to a status via [`core_status`]).
    Core(asset_studio_core::Error),
    /// A bad request payload (always 400).
    BadRequest(String),
}

impl ApiError {
    /// The HTTP status to answer with.
    fn status(&self) -> Status {
        match self {
            ApiError::Core(e) => core_status(e),
            ApiError::BadRequest(_) => Status::BAD_REQUEST,
        }
    }

    /// The client-facing error message.
    fn message(&self) -> String {
        match self {
            ApiError::Core(e) => e.to_string(),
            ApiError::BadRequest(msg) => msg.clone(),
        }
    }
}

impl From<asset_studio_core::Error> for ApiError {
    fn from(e: asset_studio_core::Error) -> Self {
        ApiError::Core(e)
    }
}

impl From<asset_studio_library::StoreError> for ApiError {
    fn from(e: asset_studio_library::StoreError) -> Self {
        match e {
            asset_studio_library::StoreError::Core(core) => ApiError::Core(core),
            other => ApiError::BadRequest(other.to_string()),
        }
    }
}

/// Map a domain error to an HTTP status.
fn core_status(e: &asset_studio_core::Error) -> Status {
    match e {
        asset_studio_core::Error::UnknownAsset(_)
        | asset_studio_core::Error::DeletedAsset(_)
        | asset_studio_core::Error::UnknownServer(_)
        | asset_studio_core::Error::UnknownTask { .. } => Status::NOT_FOUND,
        asset_studio_core::Error::InvalidInput { .. }
        | asset_studio_core::Error::UnacceptableInput { .. }
        | asset_studio_core::Error::MissingInput(_)
        | asset_studio_core::Error::InvalidAssetKind(_)
        | asset_studio_core::Error::InvalidRating
        | asset_studio_core::Error::InvalidTag(_) => Status::BAD_REQUEST,
        asset_studio_core::Error::DuplicateIdempotencyKey
        | asset_studio_core::Error::CannotCancel
        | asset_studio_core::Error::CannotRetry => Status::CONFLICT,
        asset_studio_core::Error::JobTerminated(_, _) => Status::CONFLICT,
        _ => Status::INTERNAL_SERVER_ERROR,
    }
}

/// A genuine internal failure. Returns a 500 and closes the connection when
/// propagated out of [`Studio::call`].
#[derive(Debug, thiserror::Error)]
pub enum ServerHandlerError {
    #[error("store error: {0}")]
    Store(asset_studio_library::StoreError),
}

/// Shared application state for the API handler.
///
/// `library` is guarded by a `std::sync::Mutex` because [`Library`] (its
/// `rusqlite` connection) is `!Sync`; all library operations are synchronous
/// and short, so the guard never spans an `.await`. `launcher` is already
/// internally synchronized.
pub struct App {
    pub library: Arc<Mutex<Library>>,
    pub launcher: Arc<Launcher>,
}

impl App {
    pub fn new(library: Library, launcher: Launcher) -> Self {
        Self {
            library: Arc::new(Mutex::new(library)),
            launcher: Arc::new(launcher),
        }
    }
}

/// The HTTP API handler.
#[derive(Clone)]
pub struct Studio {
    pub app: Arc<App>,
}

impl Studio {
    pub fn new(app: Arc<App>) -> Self {
        Self { app }
    }

    /// Build a JSON response.
    fn json(status: Status, value: serde_json::Value) -> Response<BufferedBody<Vec<u8>>> {
        let body = serde_json::to_vec_pretty(&value).unwrap_or_default();
        let mut response = Response::new(status, BufferedBody::from(body));
        let _ = response
            .metadata
            .fields
            .add("Content-Type", "application/json");
        response
    }

    /// Build a JSON error response from an [`ApiError`].
    fn error_response(err: &ApiError) -> Response<BufferedBody<Vec<u8>>> {
        Self::json(
            err.status(),
            serde_json::json!({ "error": err.message() }),
        )
    }

    fn not_found(msg: &str) -> Response<BufferedBody<Vec<u8>>> {
        Self::json(Status::NOT_FOUND, serde_json::json!({ "error": msg }))
    }

    /// Route a parsed request to its handler.
    ///
    /// Client-side failures are returned as `Err(ApiError)` (converted to a
    /// 4xx JSON response by [`Studio::call`]). Only genuine store I/O failures
    /// bubble up as a handler `Err` and close the connection.
    fn route(
        &self,
        method: &str,
        path: &str,
        query: &str,
        segments: &[&str],
        raw_body: &[u8],
    ) -> Result<Response<BufferedBody<Vec<u8>>>, ApiError> {
        match (method, segments) {
            ("GET", []) => Ok(Self::json(
                Status::OK,
                serde_json::json!({
                    "name": "asset-studio",
                    "status": "ok",
                    "servers": asset_studio_models::all_servers()
                        .iter()
                        .map(|s| serde_json::json!({ "id": s.id, "title": s.title, "url": s.url() }))
                        .collect::<Vec<_>>(),
                }),
            )),
            ("GET", ["servers"]) => {
                let lib = self.app.library.lock().unwrap();
                for s in asset_studio_models::all_servers() {
                    lib.upsert_server(&s.info())?;
                }
                let servers = lib.list_servers()?;
                Ok(Self::json(
                    Status::OK,
                    serde_json::json!({
                        "servers": servers
                            .iter()
                            .map(|s| serde_json::to_value(s).unwrap())
                            .collect::<Vec<_>>(),
                    }),
                ))
            }
            ("GET", ["assets"]) => {
                let kind = opt_query(query, "kind");
                let assets = self
                    .app
                    .library
                    .lock()
                    .unwrap()
                    .list_assets(kind.as_deref(), 100)?;
                Ok(Self::json(
                    Status::OK,
                    serde_json::json!({
                        "assets": assets.iter().map(|a| serde_json::to_value(a).unwrap()).collect::<Vec<_>>(),
                    }),
                ))
            }
            ("GET", ["assets", id]) => {
                let asset = self
                    .app
                    .library
                    .lock()
                    .unwrap()
                    .get_asset(&asset_studio_core::AssetId(id.to_string()))?
                    .ok_or(asset_studio_core::Error::UnknownAsset(id.to_string()))?;
                Ok(Self::json(Status::OK, serde_json::json!({ "asset": asset })))
            }
            ("GET", ["assets", id, "download"]) => {
                let a = self
                    .app
                    .library
                    .lock()
                    .unwrap()
                    .get_asset(&asset_studio_core::AssetId(id.to_string()))?
                    .ok_or(asset_studio_core::Error::UnknownAsset(id.to_string()))?;
                let bytes = self
                    .app
                    .library
                    .lock()
                    .unwrap()
                    .read_asset_bytes(&asset_studio_core::AssetId(id.to_string()))?;
                let mut response = Response::new(Status::OK, BufferedBody::from(bytes));
                let _ = response.metadata.fields.add("Content-Type", &a.mime);
                let _ = response
                    .metadata
                    .fields
                    .add("Content-Length", &a.size_bytes.to_string());
                Ok(response)
            }
            ("POST", ["assets", "import"]) => {
                let kind = opt_query(query, "kind").unwrap_or_else(|| "image".to_string());
                let title = opt_query(query, "title");
                let mut tags: Vec<String> = opt_query(query, "tags")
                    .map(|t| t.split(',').map(str::trim).filter(|t| !t.is_empty()).map(String::from).collect())
                    .unwrap_or_default();
                if let Some(source) = opt_query(query, "derived_from") {
                    // The source must exist: an agent's edit names what it edited.
                    self.app
                        .library
                        .lock()
                        .unwrap()
                        .get_asset(&asset_studio_core::AssetId(source.clone()))?
                        .ok_or(asset_studio_core::Error::UnknownAsset(source.clone()))?;
                    tags.push(format!("derived:{source}"));
                }
                import_asset(&self.app.library, &kind, &title, tags, raw_body.to_vec())
            }
            ("PUT", ["assets", id, "rating"]) => {
                let rating: Option<u8> = serde_json::from_slice(raw_body)
                    .map_err(|e| ApiError::BadRequest(format!("bad json: {e}")))?;
                self.app
                    .library
                    .lock()
                    .unwrap()
                    .set_rating(&asset_studio_core::AssetId(id.to_string()), rating)?;
                Ok(Self::json(Status::OK, serde_json::json!({ "ok": true })))
            }
            ("DELETE", ["assets", id]) => {
                self.app
                    .library
                    .lock()
                    .unwrap()
                    .delete_asset(
                        &asset_studio_core::AssetId(id.to_string()),
                        asset_studio_core::time::now(),
                    )?;
                Ok(Self::json(Status::OK, serde_json::json!({ "ok": true })))
            }
            ("GET", ["jobs"]) => {
                let state = opt_query(query, "state");
                let jobs = self
                    .app
                    .library
                    .lock()
                    .unwrap()
                    .list_jobs(state.as_deref(), 100)?;
                Ok(Self::json(
                    Status::OK,
                    serde_json::json!({
                        "jobs": jobs.iter().map(|j| serde_json::to_value(j).unwrap()).collect::<Vec<_>>(),
                    }),
                ))
            }
            ("POST", ["jobs"]) => submit_job(&self.app, raw_body),
            ("POST", ["pipelines", "text-to-3d"]) => text_to_3d(&self.app, raw_body),
            ("GET", ["jobs", id]) => {
                let job = self
                    .app
                    .library
                    .lock()
                    .unwrap()
                    .get_job(&asset_studio_core::JobId(id.to_string()))?
                    .ok_or(asset_studio_core::Error::UnknownAsset(id.to_string()))?;
                Ok(Self::json(Status::OK, serde_json::json!({ "job": job })))
            }
            ("POST", ["jobs", id, "cancel"]) => cancel_job(&self.app.library, id),
            ("POST", ["jobs", id, "retry"]) => retry_job(&self.app.library, id),
            ("GET", ["health"]) => Ok(Self::json(Status::OK, serde_json::json!({ "status": "ok" }))),
            _ => Ok(Self::not_found(&format!("no route for {method} {path}"))),
        }
    }
}

impl Handler for Studio {
    type Error = ServerHandlerError;
    type Body<B: ReadableBody> = BufferedBody<Vec<u8>>;

    async fn call<B: ReadableBody, I: InterimHandler>(
        &self,
        request: Request<B>,
        _interim: &mut InterimResponses<I>,
        cancellation: &Cancellation,
    ) -> Result<Response<Self::Body<B>>, Self::Error> {
        // Capture routing metadata *before* consuming the request body.
        let method = request.metadata.method;
        let path = request
            .metadata
            .path
            .as_ref()
            .map(|p| p.as_str().to_string())
            .unwrap_or_default();
        let (path, query) = path.split_once('?').unwrap_or((path.as_str(), ""));
        let query = query.to_string();
        let segments: Vec<&str> = path
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        // Read the (usually-empty) request body to completion.
        let raw_body = body_to_bytes(request.body, cancellation).await;

        // Route the request. Routing errors become 4xx JSON responses; genuine
        // store I/O failures bubble up as a handler `Err` and close the
        // connection (which `serve_connection` then surfaces as a 500).
        let result = self.route(method.as_str(), path, &query, &segments, &raw_body);
        match result {
            Ok(response) => Ok(response),
            Err(api) => Ok(Self::error_response(&api)),
        }
    }
}

/// Import an asset from raw bytes (used by `POST /assets/import`).
fn import_asset(
    library: &Arc<Mutex<Library>>,
    kind: &str,
    title: &Option<String>,
    tags: Vec<String>,
    bytes: Vec<u8>,
) -> Result<Response<BufferedBody<Vec<u8>>>, ApiError> {
    let kind: asset_studio_core::AssetKind = kind.parse()?;
    let accepted = kind.accepted_mime();
    let mime = match sniff_mime(&bytes) {
        Some(mime) if accepted.contains(&mime) => mime.to_string(),
        _ if accepted.contains(&"application/octet-stream") => "application/octet-stream".to_string(),
        sniffed => {
            return Err(ApiError::BadRequest(format!(
                "a {} asset must be one of {} (the file looks like {})",
                kind.as_str(),
                accepted.join(", "),
                sniffed.unwrap_or("unknown data")
            )))
        }
    };
    let asset = asset_studio_core::Asset {
        id: new_asset_id(),
        kind,
        mime,
        title: title.clone(),
        size_bytes: bytes.len() as u64,
        blob_sha256: String::new(),
        rating: None,
        tags,
        producer: None,
        created_at: asset_studio_core::time::now(),
        deleted_at: None,
    };
    asset.validate_tags()?;
    let stored = library.lock().unwrap().upsert_asset(&asset, &bytes)?;
    Ok(Studio::json(
        Status::CREATED,
        serde_json::json!({ "asset": stored }),
    ))
}

/// Submit a job from a JSON payload.
fn submit_job(app: &Arc<App>, raw: &[u8]) -> Result<Response<BufferedBody<Vec<u8>>>, ApiError> {
    let req: serde_json::Value = serde_json::from_slice(raw)
        .map_err(|e| ApiError::BadRequest(format!("bad json: {e}")))?;
    let job = build_job(&req)?;
    app.library.lock().unwrap().insert_job(&job)?;
    Ok(Studio::json(
        Status::CREATED,
        serde_json::json!({ "job": job }),
    ))
}

/// Submit the text-to-3D chain as three dependent jobs:
///
/// 1. `zimage` `text-to-image`: the prompt as one object on a plain
///    background (`object_reference`);
/// 2. `birefnet` `image-to-image`: its first picture, cut out;
/// 3. `trellis2_3d` `image-to-3d`: the cut-out as a textured GLB.
///
/// Body: `{prompt, seed?, height_m?, image_params?, cutout_params?,
/// model_params?}`. Each later job takes output 0 of the one before, so the
/// orchestrators run them in order and hand the GPU along. Returns
/// `{jobs: [picture, cutout, model], model_job}`; the model job's output is
/// the GLB.
fn text_to_3d(app: &Arc<App>, raw: &[u8]) -> Result<Response<BufferedBody<Vec<u8>>>, ApiError> {
    let req: serde_json::Value = serde_json::from_slice(raw)
        .map_err(|e| ApiError::BadRequest(format!("bad json: {e}")))?;
    let prompt = req
        .get("prompt")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .ok_or_else(|| ApiError::BadRequest("prompt: a non-empty string is required".into()))?;
    let seeds = req.get("seed").map(|seed| serde_json::json!([seed]));
    let object = |value: Option<&serde_json::Value>| -> Result<serde_json::Map<String, serde_json::Value>, ApiError> {
        match value {
            None | Some(serde_json::Value::Null) => Ok(serde_json::Map::new()),
            Some(serde_json::Value::Object(map)) => Ok(map.clone()),
            Some(_) => Err(ApiError::BadRequest("*_params: expected an object".into())),
        }
    };
    let mut image_params = object(req.get("image_params"))?;
    image_params.entry("object_reference").or_insert(true.into());
    let mut model_params = object(req.get("model_params"))?;
    if let Some(height) = req.get("height_m") {
        model_params.insert("height_m".into(), height.clone());
    }

    let picture = build_job(&serde_json::json!({
        "server": "zimage", "task": "text-to-image", "prompt": prompt,
        "params": image_params, "count": 1, "seeds": seeds,
    }))?;
    let cutout = build_job(&serde_json::json!({
        "server": "birefnet", "task": "image-to-image",
        "params": object(req.get("cutout_params"))?,
        "inputs": { "image": { "job": picture.id.0, "output": 0 } },
    }))?;
    let model = build_job(&serde_json::json!({
        "server": "trellis2_3d", "task": "image-to-3d", "params": model_params, "seeds": seeds,
        "inputs": { "image": { "job": cutout.id.0, "output": 0 } },
    }))?;
    {
        let library = app.library.lock().unwrap();
        for job in [&picture, &cutout, &model] {
            library.insert_job(job)?;
        }
    }
    Ok(Studio::json(
        Status::CREATED,
        serde_json::json!({ "jobs": [picture, cutout, model], "model_job": model.id }),
    ))
}

/// Build (and validate) a queued job from a submit payload.
fn build_job(req: &serde_json::Value) -> Result<asset_studio_core::Job, ApiError> {
    let server = req.get("server").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let task = req.get("task").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let prompt = req.get("prompt").and_then(|v| v.as_str()).map(|s| s.to_string());
    let params = req.get("params").cloned().unwrap_or(serde_json::Value::Null);
    let count = req.get("count").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let inputs_raw = req.get("inputs").cloned().unwrap_or(serde_json::Value::Null);
    let idempotency_key = req
        .get("idempotency_key")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Validate the server + task exist in the catalog.
    let server_info = asset_studio_models::find_server(&server)
        .ok_or(asset_studio_core::Error::UnknownServer(server.clone()))?;
    let task_info = server_info
        .task(&task)
        .ok_or(asset_studio_core::Error::UnknownTask {
            server: server.clone(),
            task: task.clone(),
        })?;

    // Parse inputs: {role: {asset: id} | {job: id, output: n}}.
    let mut inputs = std::collections::BTreeMap::new();
    if let serde_json::Value::Object(map) = inputs_raw {
        for (role, v) in map {
            let role_id = role.clone();
            if let Some(aid) = v.get("asset").and_then(|x| x.as_str()) {
                inputs.insert(
                    role,
                    asset_studio_core::InputRef::Asset(asset_studio_core::AssetRef {
                        id: asset_studio_core::AssetId(aid.to_string()),
                        role: role_id,
                    }),
                );
            } else if let (Some(jid), Some(idx)) = (
                v.get("job").and_then(|x| x.as_str()),
                v.get("output").and_then(|x| x.as_u64()),
            ) {
                inputs.insert(
                    role,
                    asset_studio_core::InputRef::JobOutput(asset_studio_core::Dependency {
                        job_id: asset_studio_core::JobId(jid.to_string()),
                        output_index: idx as usize,
                    }),
                );
            }
        }
    }

    let now = asset_studio_core::time::now();
    let job = asset_studio_core::Job {
        id: job_id(),
        server: server.clone(),
        task: task.clone(),
        state: JobState::Queued,
        prompt,
        params,
        inputs,
        count,
        seeds: req.get("seeds").and_then(|v| v.as_array()).map(|a| {
            a.iter().filter_map(|x| x.as_u64()).collect::<Vec<_>>()
        }),
        idempotency_key,
        outputs: vec![],
        error: None,
        created_at: now,
        updated_at: now,
        retry_of: None,
    };

    job.validate_inputs(task_info)?;
    if task_info.uses_prompt && job.prompt.as_deref().map_or(true, |p| p.trim().is_empty()) {
        return Err(ApiError::BadRequest(format!("{server}/{task} needs a prompt")));
    }
    Ok(job)
}

/// Cancel a job (only legal while queued/waiting/running).
fn cancel_job(
    library: &Arc<Mutex<Library>>,
    id: &str,
) -> Result<Response<BufferedBody<Vec<u8>>>, ApiError> {
    let mut job = library
        .lock()
        .unwrap()
        .get_job(&asset_studio_core::JobId(id.to_string()))?
        .ok_or(asset_studio_core::Error::UnknownAsset(id.to_string()))?;
    job.transition(JobState::Cancelled, asset_studio_core::time::now())?;
    library.lock().unwrap().update_job(&job)?;
    Ok(Studio::json(Status::OK, serde_json::json!({ "job": job })))
}

/// Retry a failed job by creating a new job that references the same inputs.
fn retry_job(
    library: &Arc<Mutex<Library>>,
    id: &str,
) -> Result<Response<BufferedBody<Vec<u8>>>, ApiError> {
    let now = asset_studio_core::time::now();
    let orig = library
        .lock()
        .unwrap()
        .get_job(&asset_studio_core::JobId(id.to_string()))?
        .ok_or(asset_studio_core::Error::UnknownAsset(id.to_string()))?;
    if orig.state != JobState::Failed {
        return Err(asset_studio_core::Error::CannotRetry.into());
    }
    let mut retry = orig.clone();
    retry.id = job_id();
    retry.state = JobState::Queued;
    retry.outputs = vec![];
    retry.error = None;
    retry.retry_of = Some(asset_studio_core::JobId(id.to_string()));
    retry.created_at = now;
    retry.updated_at = now;
    library.lock().unwrap().insert_job(&retry)?;
    Ok(Studio::json(
        Status::CREATED,
        serde_json::json!({ "job": retry }),
    ))
}

// ---- helpers --------------------------------------------------------------

/// Read a request body to completion.
async fn body_to_bytes<B: ReadableBody>(mut body: B, cancellation: &Cancellation) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        match body.read(cancellation).await {
            Ok(BodyFrame::Data { data, .. }) => out.extend_from_slice(data.as_ref()),
            Ok(BodyFrame::Headers { .. }) => {}
            Ok(BodyFrame::End) => break,
            Err(_) => break, // best-effort: treat the rest as empty
        }
    }
    out
}

/// Extract an optional query parameter from a query string, percent-decoded
/// (`+` is a space, as in form encoding).
fn opt_query(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (percent_decode(k) == key).then(|| percent_decode(v))
    })
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() && text.is_char_boundary(i + 3) => {
                match u8::from_str_radix(&text[i + 1..i + 3], 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            byte => out.push(byte),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The MIME type of a file, from its leading bytes.
fn sniff_mime(bytes: &[u8]) -> Option<&'static str> {
    match bytes {
        [0x89, b'P', b'N', b'G', ..] => Some("image/png"),
        [0xFF, 0xD8, 0xFF, ..] => Some("image/jpeg"),
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P', ..] => Some("image/webp"),
        [b'g', b'l', b'T', b'F', ..] => Some("model/gltf-binary"),
        _ => None,
    }
}

fn job_id() -> asset_studio_core::JobId {
    asset_studio_core::JobId(uuid::Uuid::new_v4().to_string())
}

fn new_asset_id() -> asset_studio_core::AssetId {
    asset_studio_core::AssetId(uuid::Uuid::new_v4().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_query_parses() {
        assert_eq!(opt_query("state=queued&limit=5", "state"), Some("queued".into()));
        assert_eq!(opt_query("state=queued&limit=5", "limit"), Some("5".into()));
        assert_eq!(opt_query("state=queued", "missing"), None);
        assert_eq!(opt_query("", "state"), None);
        assert_eq!(
            opt_query("title=Cute+anime+snail+%E2%80%94+v2&x=1", "title"),
            Some("Cute anime snail — v2".into())
        );
        assert_eq!(opt_query("t=100%25", "t"), Some("100%".into()));
        assert_eq!(opt_query("t=bad%2", "t"), Some("bad%2".into()));
        assert_eq!(opt_query("t=bad%zz", "t"), Some("bad%zz".into()));
    }

    #[test]
    fn sniffs_image_and_model_types() {
        assert_eq!(sniff_mime(b"\x89PNG\r\n\x1a\n"), Some("image/png"));
        assert_eq!(sniff_mime(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("image/jpeg"));
        assert_eq!(sniff_mime(b"RIFF\0\0\0\0WEBPVP8 "), Some("image/webp"));
        assert_eq!(sniff_mime(b"glTF\x02\0\0\0"), Some("model/gltf-binary"));
        assert_eq!(sniff_mime(b"hello"), None);
    }

    #[test]
    fn core_status_mapping() {
        assert_eq!(
            core_status(&asset_studio_core::Error::UnknownAsset("a".into())),
            Status::NOT_FOUND
        );
        assert_eq!(
            core_status(&asset_studio_core::Error::DuplicateIdempotencyKey),
            Status::CONFLICT
        );
        assert_eq!(
            core_status(&asset_studio_core::Error::MissingInput("r".into())),
            Status::BAD_REQUEST
        );
    }
}
