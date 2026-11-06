//! The job orchestrator: the async work loop that turns queued jobs into
//! generated assets for one model server.
//!
//! Model servers speak the *synchronous* contract v1: `POST /v1/generate`
//! blocks until the generation finishes and returns the outputs inline (as
//! base64), and `GET /v1/info` reports readiness (`loading | ready | busy |
//! error`). There is no job polling on the model side — one request *is* one
//! job.
//!
//! Servers run **on demand** (see [`run`]): a server's process starts when
//! one of its jobs is ready (its dependencies have succeeded) and the GPU
//! slot and enough GPU memory are free, and stops when it has been idle for
//! a while, or at once when idle and another GPU server has work. Jobs that
//! reference another job's output therefore form a pipeline — e.g. prompt →
//! picture (Z-Image) → cut-out (BiRefNet) → textured GLB (TRELLIS.2) — that
//! hands the single GPU from one model to the next.
//!
//! Within a session the job loop and the HTTP client's driver run
//! concurrently in the same async scope. Pacing uses the scheduler's
//! [`Clock`](base_scheduler::Clock); process reaping uses the launcher's
//! non-blocking [`poll`](asset_studio_launcher::Launcher::poll).
use std::collections::BTreeMap;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use asset_studio_core::{Asset, AssetId, AssetKind, InputRef, Job, JobId, JobState};
use asset_studio_launcher::ServerStatus;
use asset_studio_library::Library;
use asset_studio_models::{GenerateResult, ModelClient};
use base_scheduler::{
    Cancellation, Clock, SocketClose, SocketConnect, SocketOpen, SocketSetOption,
    SocketStreamReading, SocketStreamWriting, TcpNoDelay,
};

use crate::App;

/// Maximum times a job is re-queued after a *retryable* model error before
/// it is failed (e.g. repeated `busy` or transient transport drops).
const MAX_RETRIES: u32 = 3;

/// Scheduler capabilities required to run the orchestrator (a
/// [`base_scheduler::Clock`] plus the socket caps
/// [`ModelClient::connect`](asset_studio_models::ModelClient::connect) needs).
pub trait Scheduler:
    Clock
    + SocketOpen
    + SocketConnect
    + SocketClose
    + SocketSetOption<TcpNoDelay>
    + SocketStreamReading
    + SocketStreamWriting
    + 'static
{
}

impl<S> Scheduler for S where
    S: Clock
        + SocketOpen
        + SocketConnect
        + SocketClose
        + SocketSetOption<TcpNoDelay>
        + SocketStreamReading
        + SocketStreamWriting
        + 'static
{
}

/// How long a GPU model server stays up with no work before it is stopped
/// (`ASSET_STUDIO_IDLE_SECONDS`, default 120). It stops sooner, as soon as it
/// is idle, if another GPU server has a job waiting for the slot.
fn idle_timeout() -> Duration {
    let seconds = std::env::var("ASSET_STUDIO_IDLE_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    Duration::from_secs(seconds)
}

/// How long a model may take to report `ready` after starting
/// (`ASSET_STUDIO_LOAD_SECONDS`, default 900) before its job fails.
fn load_timeout() -> Duration {
    let seconds = std::env::var("ASSET_STUDIO_LOAD_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(900);
    Duration::from_secs(seconds)
}

const TICK: Duration = Duration::from_millis(500);

/// Run the orchestrator for one model server until `cancellation`.
///
/// The server process runs **on demand**: while no job for it is ready the
/// orchestrator only watches the queue. When one is, it waits for the GPU
/// slot and for enough free GPU memory, starts the process, connects a
/// client and runs a *session* that dispatches jobs one at a time. The
/// session ends when the server has been idle for [`idle_timeout`], or as
/// soon as it is idle while another GPU server has work waiting (so a
/// text → picture → cut-out → 3D chain hands the GPU along), or when the
/// process exits. The process is then stopped and the loop starts over, so
/// a crashed server is restarted by the next job.
pub async fn run<S: Scheduler>(
    scheduler: &S,
    app: &Arc<App>,
    server_id: &str,
    cancellation: &Cancellation,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let server = asset_studio_models::find_server(server_id)
        .ok_or_else(|| format!("unknown server {server_id}"))?;
    let addr: SocketAddr = format!("127.0.0.1:{}", server.port).parse()?;
    let mut waiting_for: Option<String> = None;

    while !cancellation.is_cancelled() {
        fail_blocked_jobs(app, server_id);
        if pick_ready_job(app, server_id).is_none() {
            waiting_for = None;
            app.waiting.lock().unwrap().remove(server_id);
            let _ = scheduler.sleep(TICK, cancellation).await;
            continue;
        }

        match start(app, &server) {
            Ok(pid) => {
                waiting_for = None;
                app.waiting.lock().unwrap().remove(server_id);
                tracing::info!(server = server_id, pid, "model server started");
            }
            Err(Start::Wait(reason)) => {
                if waiting_for.as_deref() != Some(reason.as_str()) {
                    tracing::info!(server = server_id, "job waiting: {reason}");
                    app.waiting.lock().unwrap().insert(server_id.to_string(), reason.clone());
                    waiting_for = Some(reason);
                }
                let _ = scheduler.sleep(TICK, cancellation).await;
                continue;
            }
            Err(Start::Failed(reason)) => {
                tracing::warn!(server = server_id, "cannot start model server: {reason}");
                if let Some(job) = pick_ready_job(app, server_id) {
                    fail_job(app, &job.id, format!("cannot start model server: {reason}"));
                }
                continue;
            }
        }

        let (client, driver) = match ModelClient::connect(scheduler, addr) {
            Ok(pair) => pair,
            Err(e) => {
                let _ = app.launcher.kill(server_id);
                return Err(format!("connecting to {addr}: {e}").into());
            }
        };
        let session_cancellation = cancellation.child();
        let outcome = select2(
            session(scheduler, app, &client, &server, &session_cancellation),
            driver.run(&session_cancellation),
        )
        .await;
        let _ = app.launcher.kill(server_id);
        match outcome {
            Ok(SessionEnd::Idle) => tracing::info!(server = server_id, "model server stopped (idle)"),
            Ok(SessionEnd::Yielded) => {
                tracing::info!(server = server_id, "model server stopped (GPU handed over)")
            }
            Ok(SessionEnd::Cancelled) => {}
            Ok(SessionEnd::Exited(reason)) => {
                tracing::warn!(server = server_id, "model server stopped: {reason}")
            }
            Err(Ok(())) => {}
            Err(Err(e)) => tracing::warn!(server = server_id, "client driver failed: {e}"),
        }
    }
    tracing::info!(server = server_id, "orchestrator stopped");
    Ok(())
}

/// Why a server could not be started now.
enum Start {
    /// Not now: the GPU slot or memory is taken. The job stays queued.
    Wait(String),
    /// The process could not be spawned at all.
    Failed(String),
}

/// Start `server`'s process: needs the GPU slot and `vram_mib` of free GPU memory.
fn start(app: &Arc<App>, server: &asset_studio_models::ModelServer) -> Result<u32, Start> {
    let launcher = &app.launcher;
    if server.uses_gpu {
        let holders = launcher.gpu_holders();
        if !holders.is_empty() {
            return Err(Start::Wait(format!("the GPU is in use by {}", holders.join(", "))));
        }
        if let Some(free) = asset_studio_launcher::gpu_free_mib() {
            if free < server.vram_mib {
                return Err(Start::Wait(format!(
                    "{} MiB of GPU memory free, {} MiB needed (is another program, e.g. Ollama, using the GPU?)",
                    free, server.vram_mib
                )));
            }
        }
    }
    match launcher.spawn(&server.spec()) {
        Ok(pid) => Ok(pid),
        Err(e) if matches!(e.kind(), std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::Interrupted) => {
            Err(Start::Wait(e.to_string()))
        }
        Err(e) => Err(Start::Failed(e.to_string())),
    }
}

/// Why a session ended.
enum SessionEnd {
    /// No work for [`idle_timeout`].
    Idle,
    /// Idle while another GPU server had work waiting.
    Yielded,
    /// The orchestrator is shutting down.
    Cancelled,
    /// The process exited or its state was lost.
    Exited(String),
}

/// Dispatch ready jobs to a running server until it should stop.
async fn session<S: Scheduler + ?Sized>(
    scheduler: &S,
    app: &Arc<App>,
    client: &ModelClient<'_>,
    server: &asset_studio_models::ModelServer,
    cancellation: &Cancellation,
) -> SessionEnd {
    let server_id = server.id.as_str();
    let mut idle_since = std::time::Instant::now();
    loop {
        if cancellation.is_cancelled() {
            return SessionEnd::Cancelled;
        }
        if let Some(reason) = process_gone(app, server_id) {
            fail_in_flight_jobs(app, server_id, reason.clone());
            return SessionEnd::Exited(reason);
        }
        fail_blocked_jobs(app, server_id);
        if let Some(job) = pick_ready_job(app, server_id) {
            dispatch::<S>(scheduler, app, client, server_id, &job, cancellation).await;
            idle_since = std::time::Instant::now();
            continue;
        }
        if server.uses_gpu && other_gpu_server_waiting(app, server_id) {
            return SessionEnd::Yielded;
        }
        if idle_since.elapsed() >= idle_timeout() {
            return SessionEnd::Idle;
        }
        let _ = scheduler.sleep(TICK, cancellation).await;
    }
}

/// `Some(reason)` if the server's process has exited (reaping it).
fn process_gone(app: &Arc<App>, server_id: &str) -> Option<String> {
    match app.launcher.poll(server_id) {
        Ok(ServerStatus::Exited(code)) => Some(format!("model server process exited (code {code})")),
        Ok(_) => None,
        Err(_) => Some("model server process state lost".into()),
    }
}

/// Whether another GPU server in the catalog has a job ready to run.
fn other_gpu_server_waiting(app: &Arc<App>, server_id: &str) -> bool {
    asset_studio_models::all_servers()
        .iter()
        .filter(|s| s.uses_gpu && s.id != server_id)
        .any(|s| pick_ready_job(app, &s.id).is_some())
}

/// The oldest `Queued` job for `server_id` whose dependencies have all succeeded.
fn pick_ready_job(app: &Arc<App>, server_id: &str) -> Option<Job> {
    let lib = app.library.lock().unwrap();
    let (jobs, states) = (lib.list_jobs(None, 512).ok()?, lib.job_states().ok()?);
    let states_map: BTreeMap<JobId, JobState> = states.into_iter().collect();
    // `list_jobs` is newest first.
    jobs.into_iter().rev().find(|j| {
        j.server == server_id
            && j.state == JobState::Queued
            && j
                .dependencies()
                .iter()
                .all(|d| states_map.get(&d.job_id) == Some(&JobState::Succeeded))
    })
}

/// Fail queued jobs for `server_id` whose dependency failed, was cancelled
/// or does not exist: they could never run.
fn fail_blocked_jobs(app: &Arc<App>, server_id: &str) {
    let blocked: Vec<(JobId, String)> = {
        let lib = app.library.lock().unwrap();
        let (Ok(jobs), Ok(states)) = (lib.list_jobs(Some("queued"), 512), lib.job_states()) else {
            return;
        };
        let states: BTreeMap<JobId, JobState> = states.into_iter().collect();
        jobs.into_iter()
            .filter(|j| j.server == server_id)
            .filter_map(|j| {
                j.dependencies().iter().find_map(|d| match states.get(&d.job_id) {
                    None => Some((j.id.clone(), format!("dependency {} does not exist", d.job_id))),
                    Some(JobState::Failed) | Some(JobState::Cancelled) => {
                        Some((j.id.clone(), format!("dependency {} did not succeed", d.job_id)))
                    }
                    _ => None,
                })
            })
            .collect()
    };
    for (id, reason) in blocked {
        fail_job(app, &id, reason);
    }
}

/// Drive one job through the synchronous contract:
///
/// `queued → waiting_model → (wait for /v1/info ready) → running →
/// POST /v1/generate (blocking) → store outputs → succeeded`
///
/// On a *retryable* failure the job is re-queued (retries are tracked in the
/// job's `params`); on a non-retryable failure it is marked `Failed`.
async fn dispatch<S: Scheduler + ?Sized>(
    scheduler: &S,
    app: &Arc<App>,
    client: &ModelClient<'_>,
    server_id: &str,
    job: &Job,
    cancellation: &Cancellation,
) {
    // Resolve and read the input files up front (a hard failure here is not
    // retryable: the inputs cannot change).
    let inputs = match resolve_inputs(app, job) {
        Ok(inputs) => inputs,
        Err(e) => {
            fail_job(app, &job.id, format!("inputs: {e}"));
            return;
        }
    };

    let mut working = job.clone();
    if finished_elsewhere(app, &job.id)
        || working
            .transition(JobState::WaitingModel, asset_studio_core::time::now())
            .is_err()
    {
        return;
    }
    let _ = app.library.lock().unwrap().update_job(&working);

    // Wait for the server to report `ready`.
    if !wait_for_ready::<S>(scheduler, app, client, server_id, &working, cancellation).await {
        return;
    }

    if finished_elsewhere(app, &job.id) {
        tracing::info!(job = %job.id.0, "job cancelled while waiting for the model");
        return;
    }
    if working.transition(JobState::Running, asset_studio_core::time::now()).is_err() {
        return;
    }
    let _ = app.library.lock().unwrap().update_job(&working);
    tracing::info!(job = %working.id.0, server = %working.server, "generation started");

    // The synchronous generate: blocks until the model finishes.
    let seed = working.seeds.as_ref().and_then(|s| s.first().copied());
    let result = client
        .generate(
            &working.task,
            working.prompt.as_deref(),
            &inputs,
            &model_params(&working.params),
            working.count as u64,
            seed,
            cancellation,
        )
        .await;

    if finished_elsewhere(app, &job.id) {
        tracing::info!(job = %job.id.0, "job cancelled while generating; outputs discarded");
        return;
    }
    match result {
        Ok(result) => match store_outputs(app, &working, &result) {
            Ok(asset_ids) => {
                let mut done = working;
                done.outputs = asset_ids;
                // Record the seed actually used (the server picks one if none was given).
                done.seeds = Some(
                    (0..result.outputs.len() as u64)
                        .map(|i| (result.seed + i) % (1 << 32))
                        .collect(),
                );
                if done.transition(JobState::Succeeded, asset_studio_core::time::now()).is_ok() {
                    let _ = app.library.lock().unwrap().update_job(&done);
                    tracing::info!(
                        job = %done.id.0,
                        outputs = done.outputs.len(),
                        generate_s = result.generate_s,
                        "job succeeded"
                    );
                }
            }
            Err(e) => fail_job(app, &working.id, format!("storing outputs: {e}")),
        },
        Err(e) if e.retryable() => {
            let reason = e.to_string();
            requeue(app, &working, reason);
        }
        Err(e) => fail_job(app, &working.id, format!("generation failed: {e}")),
    }
}

/// Poll `GET /v1/info` until the server reports `ready` (or `cancellation`).
///
/// `loading`/`busy` are transient; `error` and non-retryable failures are
/// terminal. On cancellation the job is re-queued so it is not lost.
///
/// Returns `true` to proceed to generation, `false` if the job was already
/// retired (failed) or re-queued by the helpers below.
async fn wait_for_ready<S: Scheduler + ?Sized>(
    scheduler: &S,
    app: &Arc<App>,
    client: &ModelClient<'_>,
    server_id: &str,
    job: &Job,
    cancellation: &Cancellation,
) -> bool {
    let started = std::time::Instant::now();
    loop {
        if cancellation.is_cancelled() {
            requeue(app, job, "cancelled while waiting for model".into());
            return false;
        }
        if let Some(reason) = process_gone(app, server_id) {
            // The session sees the exit next and restarts on demand; retry the job.
            requeue(app, job, reason);
            return false;
        }
        if started.elapsed() > load_timeout() {
            fail_job(app, &job.id, format!("model did not become ready within {:?}", load_timeout()));
            return false;
        }
        match client.info(cancellation).await {
            Ok(info) => match info.status.as_str() {
                "ready" => return true,
                "loading" | "busy" => {
                    let _ = scheduler.sleep(Duration::from_secs(1), cancellation).await;
                }
                other => {
                    let detail = info
                        .status_message
                        .unwrap_or_else(|| other.to_string());
                    fail_job(app, &job.id, format!("model server not ready: {detail}"));
                    return false;
                }
            },
            Err(e) if e.retryable() => {
                let _ = scheduler.sleep(Duration::from_secs(1), cancellation).await;
            }
            Err(e) => {
                fail_job(app, &job.id, format!("model server unreachable: {e}"));
                return false;
            }
        }
    }
}

/// Resolve a job's inputs to `(role, mime, bytes)` in role order.
fn resolve_inputs(
    app: &Arc<App>,
    job: &Job,
) -> Result<Vec<(String, String, Vec<u8>)>, Box<dyn std::error::Error>> {
    let lib = app.library.lock().unwrap();
    let lib: &Library = &lib;
    let mut inputs = Vec::new();
    for (role, input) in &job.inputs {
        let asset = resolve_input(lib, input)?
            .ok_or_else(|| format!("missing asset for role {role}"))?;
        let bytes = lib.read_asset_bytes(&asset.id)?;
        inputs.push((role.clone(), asset.mime.clone(), bytes));
    }
    Ok(inputs)
}

/// Resolve an [`InputRef`] to a concrete asset (a `JobOutput` resolves to the
/// dependency's `outputs[output_index]`). Takes an already-locked `&Library`.
fn resolve_input(
    lib: &Library,
    input: &InputRef,
) -> Result<Option<Asset>, asset_studio_library::StoreError> {
    let asset_id = match input {
        InputRef::Asset(ref_) => ref_.id.clone(),
        InputRef::JobOutput(dep) => {
            let outputs = lib.job_outputs(&dep.job_id)?;
            outputs.get(dep.output_index).cloned().ok_or(
                asset_studio_library::StoreError::Core(
                    asset_studio_core::Error::InvalidInput {
                        task: String::new(),
                        detail: "dependency has no outputs yet".into(),
                    },
                ),
            )?
        }
    };
    lib.get_asset(&asset_id)
}

/// Store each generated output as a library asset; returns the new asset ids
/// in output order.
fn store_outputs(
    app: &Arc<App>,
    job: &Job,
    result: &GenerateResult,
) -> Result<Vec<AssetId>, Box<dyn std::error::Error>> {
    let now = asset_studio_core::time::now();
    let mut ids = Vec::new();
    let lib = app.library.lock().unwrap();
    let base = title_base(&lib, job);
    let many = result.outputs.len() > 1;
    for (i, out) in result.outputs.iter().enumerate() {
        let id = AssetId(uuid::Uuid::new_v4().to_string());
        let asset = Asset {
            id: id.clone(),
            kind: asset_kind_for_mime(&out.mime),
            mime: out.mime.clone(),
            title: Some(if many {
                format!("{base} · {} #{}", job.task, i + 1)
            } else {
                format!("{base} · {}", job.task)
            }),
            size_bytes: out.data.len() as u64,
            blob_sha256: String::new(), // filled in by upsert
            rating: None,
            tags: vec!["generated".to_string()],
            producer: Some(job.id.0.clone()),
            created_at: now,
            deleted_at: None,
        };
        let stored = lib.upsert_asset(&asset, &out.data)?;
        ids.push(stored.id);
    }
    Ok(ids)
}

/// What a job's outputs are about: its prompt, else the base title of its
/// first input (the part before " · ", so a chain keeps the original name),
/// else the task. At most 80 characters.
fn title_base(lib: &Library, job: &Job) -> String {
    let from_input = || {
        let input = job.inputs.values().next()?;
        let asset = resolve_input(lib, input).ok()??;
        let title = asset.title?;
        Some(title.split(" · ").next().unwrap_or(&title).to_string())
    };
    let base = job
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(String::from)
        .or_else(from_input)
        .unwrap_or_else(|| job.task.clone());
    if base.chars().count() > 80 {
        format!("{}…", base.chars().take(79).collect::<String>())
    } else {
        base
    }
}

/// Map an output MIME type to the closest [`AssetKind`].
fn asset_kind_for_mime(mime: &str) -> AssetKind {
    if mime.starts_with("image/") {
        AssetKind::Image
    } else if mime.starts_with("model/") {
        AssetKind::Mesh
    } else if mime.starts_with("video/") {
        AssetKind::Video
    } else if mime.starts_with("audio/") {
        AssetKind::Audio
    } else if mime.starts_with("text/") {
        AssetKind::Text
    } else if mime == "application/octet-stream" {
        AssetKind::Motion
    } else {
        AssetKind::Image
    }
}

/// Re-queue a job with one retry recorded in `params["retries"]` (and the
/// last reason in `params["last_error"]`); once the ceiling is hit, fail it.
fn requeue(app: &Arc<App>, job: &Job, reason: String) {
    // Re-read: the job may have been cancelled while it was in flight.
    let Some(mut working) = app.library.lock().unwrap().get_job(&job.id).ok().flatten() else {
        return;
    };
    if working.state.is_terminal() {
        return;
    }
    let retries: u32 = working
        .params
        .get(RETRIES_KEY)
        .and_then(|r| r.as_u64())
        .unwrap_or(0) as u32;
    if retries + 1 > MAX_RETRIES {
        fail_job(app, &job.id, format!("retries exhausted; last: {reason}"));
        return;
    }
    relax_to_state(&mut working, JobState::Queued);
    if !working.params.is_object() {
        working.params = serde_json::json!({});
    }
    if let Some(m) = working.params.as_object_mut() {
        m.insert(RETRIES_KEY.into(), (retries + 1).into());
        m.insert(LAST_ERROR_KEY.into(), reason.clone().into());
    }
    let _ = app.library.lock().unwrap().update_job(&working);
    tracing::warn!(job = %job.id.0, retries = retries + 1, "job re-queued: {reason}");
}

/// The studio's own bookkeeping in a job's `params`: keys starting with `_`,
/// never sent to the model server (which would reject unknown parameters).
const RETRIES_KEY: &str = "_retries";
const LAST_ERROR_KEY: &str = "_last_error";

/// A job's params as the model server should see them: without the studio's
/// `_`-prefixed bookkeeping keys.
fn model_params(params: &serde_json::Value) -> serde_json::Value {
    match params {
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .filter(|(k, _)| !k.starts_with('_'))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Whether the job was cancelled (or otherwise finished) since it was picked.
fn finished_elsewhere(app: &Arc<App>, id: &JobId) -> bool {
    match app.library.lock().unwrap().get_job(id) {
        Ok(Some(job)) => job.state.is_terminal(),
        _ => true,
    }
}

/// Move a non-terminal job to `next`, bypassing the strict state machine when
/// needed (used for re-queuing after a retryable failure; the strict machine
/// has no `running -> queued` edge). Terminal jobs are left untouched.
fn relax_to_state(job: &mut Job, next: JobState) {
    if job.state.is_terminal() {
        return;
    }
    if job.transition(next, asset_studio_core::time::now()).is_err() {
        job.state = next;
        job.updated_at = asset_studio_core::time::now();
    }
}

/// Mark `job` as `Failed` with a human-readable reason.
fn fail_job(app: &Arc<App>, id: &JobId, reason: String) {
    let mut failed = {
        let lib = app.library.lock().unwrap();
        match lib.get_job(id) {
            Ok(Some(job)) => job,
            _ => return,
        }
    };
    if failed.state.is_terminal() {
        return;
    }
    relax_to_state(&mut failed, JobState::Failed);
    failed.error = Some(reason.clone());
    let _ = app.library.lock().unwrap().update_job(&failed);
    tracing::warn!(job = %id.0, "job failed: {reason}");
}

/// Mark `server_id`'s in-flight jobs (waiting for the model, or running) as
/// `Failed`; queued jobs stay queued and restart the server.
fn fail_in_flight_jobs(app: &Arc<App>, server_id: &str, reason: String) {
    let jobs = {
        let lib = app.library.lock().unwrap();
        lib.list_jobs(None, 512).unwrap_or_default()
    };
    for mut job in jobs.into_iter().filter(|j| j.server == server_id) {
        if !matches!(job.state, JobState::WaitingModel | JobState::Running) {
            continue;
        }
        relax_to_state(&mut job, JobState::Failed);
        job.error = Some(reason.clone());
        let _ = app.library.lock().unwrap().update_job(&job);
        tracing::warn!(job = %job.id.0, "job failed: {reason}");
    }
}

/// Run `a` and `b` concurrently and return the output of whichever completes
/// first, dropping the other. `Ok` if `a` won, `Err` if `b` won.
async fn select2<A, B>(a: A, b: B) -> std::result::Result<A::Output, B::Output>
where
    A: Future,
    B: Future,
{
    let mut a = Box::pin(a);
    let mut b = Box::pin(b);
    futures_lite::future::poll_fn(|cx: &mut Context<'_>| {
        if let Poll::Ready(v) = a.as_mut().poll(cx) {
            return Poll::Ready(Ok(v));
        }
        if let Poll::Ready(v) = b.as_mut().poll(cx) {
            return Poll::Ready(Err(v));
        }
        Poll::Pending
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use asset_studio_core::AssetRef;

    fn job(prompt: Option<&str>, input: Option<&AssetId>) -> Job {
        let mut inputs = BTreeMap::new();
        if let Some(id) = input {
            inputs.insert(
                "image".to_string(),
                InputRef::Asset(AssetRef { id: id.clone(), role: "image".into() }),
            );
        }
        Job {
            id: JobId::from("j"),
            server: "birefnet".into(),
            task: "image-to-image".into(),
            state: JobState::Queued,
            prompt: prompt.map(String::from),
            params: serde_json::Value::Null,
            inputs,
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
    fn bookkeeping_keys_are_not_sent_to_the_model() {
        let params = serde_json::json!({"steps": 12, "_retries": 2, "_last_error": "x"});
        assert_eq!(model_params(&params), serde_json::json!({"steps": 12}));
        assert_eq!(model_params(&serde_json::Value::Null), serde_json::Value::Null);
    }

    #[test]
    fn output_titles_follow_the_prompt_through_a_chain() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::open(dir.path()).unwrap();
        let picture = Asset {
            id: AssetId::from("a1"),
            kind: AssetKind::Image,
            mime: "image/png".into(),
            title: Some("a small mossy boulder · text-to-image".into()),
            size_bytes: 3,
            blob_sha256: String::new(),
            rating: None,
            tags: vec![],
            producer: None,
            created_at: 0,
            deleted_at: None,
        };
        let stored = lib.upsert_asset(&picture, b"png").unwrap();
        assert_eq!(title_base(&lib, &job(None, Some(&stored.id))), "a small mossy boulder");
        assert_eq!(title_base(&lib, &job(Some("  a red mushroom "), None)), "a red mushroom");
        assert_eq!(title_base(&lib, &job(None, None)), "image-to-image");
        let long = "x".repeat(200);
        assert_eq!(title_base(&lib, &job(Some(&long), None)).chars().count(), 80);
    }
}
