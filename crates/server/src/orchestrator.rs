//! The job orchestrator: the async work loop that turns queued jobs into
//! generated assets for one model server.
//!
//! Model servers speak the *synchronous* contract v1: `POST /v1/generate`
//! blocks until the generation finishes and returns the outputs inline (as
//! base64), and `GET /v1/info` reports readiness (`loading | ready | busy |
//! error`). There is no job polling on the model side — one request *is* one
//! job.
//!
//! For a given model server, [`run`](Self::run) does:
//!
//! 1. Starts the model server process (via the [`Launcher`], acquiring the
//!    GPU slot if it's a GPU server) — or adopts it if it's already running.
//! 2. Connects a typed [`ModelClient`](asset_studio_models::ModelClient).
//! 3. Drives the client's driver and a job pump **concurrently**:
//!    - the pump picks at most one `Queued` job whose dependencies are
//!      satisfied, waits for the server to report `ready`, runs the
//!      synchronous `generate`, stores every returned output as a library
//!      asset, and marks the job `Succeeded`;
//!    - it reaps the server process; when it exits, the in-flight jobs are
//!      failed and `run` returns (the caller decides whether to restart).
//!
//! The pump and driver run in the *same* async scope (avoiding the
//! `'static`-task spawn that would hit the scheduler-ownership lifetime
//! boundary). Pacing uses the scheduler's
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

/// Run the job pump for one model server, starting the server process and
/// connecting the client, until `cancellation` is observed.
///
/// Returns `Ok(())` on clean shutdown (cancellation). A process exit or a
/// driver failure returns `Err` describing *why* it stopped, so the caller
/// can decide whether to restart the server.
pub async fn run<S: Scheduler>(
    scheduler: &S,
    app: &Arc<App>,
    server_id: &str,
    cancellation: &Cancellation,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let server = asset_studio_models::find_server(server_id)
        .ok_or_else(|| format!("unknown server {server_id}"))?;
    let launcher = &app.launcher;

    // 1. Start the server process (acquires the GPU slot if needed) unless a
    //    previous run is still alive.
    if launcher.poll(server_id).unwrap_or(ServerStatus::Off) != ServerStatus::Running {
        let spec = server.spec();
        let pid = launcher.spawn(&spec)?;
        tracing::info!(server = server_id, ?pid, "model server started");
    } else {
        tracing::info!(server = server_id, "adopting already-running model server");
    }

    // 2. Connect the typed client.
    let addr: SocketAddr = format!("127.0.0.1:{}", server.port).parse()?;
    let (client, driver) = ModelClient::connect(scheduler, addr)?;

    // 3. Run the pump and the driver concurrently: whichever finishes first
    //    wins; the other is dropped.
    let inner = cancellation.child();
    let pump_fut = pump(scheduler, app, &client, server_id, &inner);
    let driver_fut = driver.run(cancellation);
    let outcome = select2(pump_fut, driver_fut).await;

    // Clean up the process, then report why we stopped.
    let _ = launcher.kill(server_id);

    match outcome {
        // Pump finished first.
        Ok(PumpExit::Cancelled) => {
            tracing::info!(server = server_id, "model server stopped");
            Ok(())
        }
        Ok(PumpExit::ProcessExited(code)) => Err(
            format!("model server process exited (code {code})").into(),
        ),
        Ok(PumpExit::ProcessLost) => Err("model server process state lost".into()),
        // Driver finished first.
        Err(Ok(())) => {
            tracing::info!(server = server_id, "client driver stopped (cancellation)");
            Ok(())
        }
        Err(Err(e)) => Err(format!("client driver failed: {e}").into()),
    }
}

/// The terminal outcome of a pump loop (other than external cancellation).
enum PumpExit {
    /// The process exited with a status code.
    ProcessExited(i32),
    /// The process state was lost.
    ProcessLost,
    /// The pump's own cancellation was observed (external shutdown).
    Cancelled,
}

/// The job pump loop for one server.
async fn pump<S: Scheduler + ?Sized>(
    scheduler: &S,
    app: &Arc<App>,
    client: &ModelClient<'_>,
    server_id: &str,
    cancellation: &Cancellation,
) -> PumpExit {
    const TICK: Duration = Duration::from_millis(250);
    let launcher = &app.launcher;

    loop {
        if cancellation.is_cancelled() {
            return PumpExit::Cancelled;
        }

        // Reap the process if it has exited.
        match launcher.poll(server_id) {
            Ok(ServerStatus::Exited(code)) => {
                fail_server_jobs(app, server_id, format!("server process exited (code {code})"));
                return PumpExit::ProcessExited(code);
            }
            Ok(_) => {}
            Err(_) => {
                fail_server_jobs(app, server_id, "server process state lost".into());
                return PumpExit::ProcessLost;
            }
        }

        if let Some(job) = pick_ready_job(app, server_id) {
            dispatch::<S>(scheduler, app, client, &job, cancellation).await;
        }

        let _ = scheduler.sleep(TICK, cancellation).await;
    }
}

/// The one `Queued` job for `server_id` whose dependencies are all satisfied,
/// if any (jobs are persisted in creation order, so first is the oldest).
fn pick_ready_job(app: &Arc<App>, server_id: &str) -> Option<Job> {
    let lib = app.library.lock().unwrap();
    let (jobs, states) = (lib.list_jobs(None, 512).ok()?, lib.job_states().ok()?);
    let states_map: BTreeMap<JobId, JobState> = states.into_iter().collect();
    jobs.into_iter()
        .find(|j| {
            j.server == server_id
                && j.state == JobState::Queued
                && j
                    .dependencies()
                    .iter()
                    .all(|d| states_map.get(&d.job_id) == Some(&JobState::Succeeded))
        })
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
    if working
        .transition(JobState::WaitingModel, asset_studio_core::time::now())
        .is_err()
    {
        return;
    }
    let _ = app.library.lock().unwrap().update_job(&working);

    // Wait for the server to report `ready`.
    if !wait_for_ready::<S>(scheduler, app, client, &working, cancellation).await {
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
            &working.params,
            working.count as u64,
            seed,
            cancellation,
        )
        .await;

    match result {
        Ok(result) => match store_outputs(app, &working, &result) {
            Ok(asset_ids) => {
                let mut done = working;
                done.outputs = asset_ids;
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
    job: &Job,
    cancellation: &Cancellation,
) -> bool {
    loop {
        if cancellation.is_cancelled() {
            requeue(app, job, "cancelled while waiting for model".into());
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
    for (i, out) in result.outputs.iter().enumerate() {
        let id = AssetId(uuid::Uuid::new_v4().to_string());
        let asset = Asset {
            id: id.clone(),
            kind: asset_kind_for_mime(&out.mime),
            mime: out.mime.clone(),
            title: Some(format!("{} #{} (job {})", job.task, i + 1, job.id.0)),
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
    let mut working = job.clone();
    let retries: u32 = working
        .params
        .get("retries")
        .and_then(|r| r.as_u64())
        .unwrap_or(0) as u32;
    if retries + 1 > MAX_RETRIES {
        fail_job(app, &job.id, format!("retries exhausted; last: {reason}"));
        return;
    }
    relax_to_state(&mut working, JobState::Queued);
    if let Some(m) = working.params.as_object_mut() {
        m.insert("retries".into(), (retries + 1).into());
        m.insert("last_error".into(), reason.clone().into());
    }
    let _ = app.library.lock().unwrap().update_job(&working);
    tracing::warn!(job = %job.id.0, retries = retries + 1, "job re-queued: {reason}");
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

/// Mark all non-terminal jobs for `server_id` as `Failed` with a reason.
fn fail_server_jobs(app: &Arc<App>, server_id: &str, reason: String) {
    let jobs = {
        let lib = app.library.lock().unwrap();
        lib.list_jobs(None, 512).unwrap_or_default()
    };
    for mut job in jobs.into_iter().filter(|j| j.server == server_id) {
        if job.state.is_terminal() {
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
