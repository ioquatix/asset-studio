//! The job orchestrator: an async pump that dispatches ready jobs to their
//! model servers and reaps running ones.
//!
//! For a given model server, [`run`](Self::run) does:
//!
//! 1. Starts the model server process (via the [`Launcher`], acquiring the
//!    GPU slot if it's a GPU server).
//! 2. Connects a typed [`ModelClient`](crate::ModelClient) to it.
//! 3. Drives the client's [`Driver`](base_http_client::Driver) and a job pump
//!    **concurrently** (zipped):
//!    - the pump marks `Queued` jobs whose dependencies are satisfied as
//!      `Running` and submits them (`POST /generate`);
//!    - it reaps the server process; on exit it fails that server's running
//!      jobs and frees the GPU slot.
//!
//! The pump and driver run in the same async scope (avoiding the
//! `'static`-task spawn that would hit the scheduler-ownership lifetime
//! boundary). Pacing uses the scheduler's
//! [`Clock`](base_scheduler::Clock); process reaping uses the launcher's
//! non-blocking [`poll`](asset_studio_launcher::Launcher::poll).
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use asset_studio_core::{Asset, InputRef, Job, JobId, JobState};
use asset_studio_launcher::{ServerStatus};
use asset_studio_library::Library;
use asset_studio_models::ModelClient;
use base_scheduler::{
    Cancellation, Clock, SocketClose, SocketConnect, SocketOpen, SocketSetOption,
    SocketStreamReading, SocketStreamWriting, TcpNoDelay,
};
use futures_lite::future::zip;

use crate::App;

/// Scheduler capabilities required to run the orchestrator (a
/// [`base_scheduler::Clock`] plus the socket caps
/// [`ModelClient::connect`](crate::ModelClient::connect) needs).
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
/// Returns `Ok(())` after cleanly stopping the server.
pub async fn run<S: Scheduler>(
    scheduler: &S,
    app: &Arc<App>,
    server_id: &str,
    cancellation: &Cancellation,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let server = asset_studio_models::find_server(server_id)
        .ok_or_else(|| format!("unknown server {server_id}"))?;
    let launcher = &app.launcher;

    // 1. Start the server process (acquires the GPU slot if needed).
    let spec = server.spec();
    let pid = launcher.spawn(&spec)?;
    tracing::info!(server = server_id, ?pid, "model server started");

    // 2. Connect the typed client.
    let addr: SocketAddr = format!("127.0.0.1:{}", server.port).parse()?;
    let (client, driver) = ModelClient::connect(scheduler, addr)?;

    // 3. Run the pump and the driver concurrently, then clean up.
    let pump_fut = pump(scheduler, app, &client, server_id, cancellation);
    let driver_fut = driver.run(cancellation);
    let _ = zip(pump_fut, driver_fut).await;

    launcher.kill(server_id)?;
    tracing::info!(server = server_id, "model server stopped");
    Ok(())
}

/// The job pump loop for one server.
async fn pump<S: Scheduler + ?Sized>(
    scheduler: &S,
    app: &Arc<App>,
    client: &ModelClient<'_>,
    server_id: &str,
    cancellation: &Cancellation,
) {
    const TICK: Duration = Duration::from_millis(250);
    let launcher = &app.launcher;

    loop {
        if cancellation.is_cancelled() {
            break;
        }

        // Reap the process if it has exited.
        match launcher.poll(server_id) {
            Ok(ServerStatus::Exited(code)) => {
                fail_server_jobs(app, server_id, format!("server process exited (code {code})"));
                break;
            }
            Ok(_) => {}
            Err(_) => {
                fail_server_jobs(app, server_id, "server process state lost".into());
                break;
            }
        }

        dispatch_ready_jobs::<S>(app, client, server_id, cancellation).await;

        let _ = scheduler.sleep(TICK, cancellation).await;
    }
}

/// Mark `Queued` jobs whose dependencies are all satisfied as `Running` and
/// submit them to the model server.
async fn dispatch_ready_jobs<S: Scheduler + ?Sized>(
    app: &Arc<App>,
    client: &ModelClient<'_>,
    server_id: &str,
    cancellation: &Cancellation,
) {
    let (jobs, states) = {
        let lib = app.library.lock().unwrap();
        (
            lib.list_jobs(None, 512).unwrap_or_default(),
            lib.job_states().unwrap_or_default(),
        )
    };
    let states_map: BTreeMap<JobId, JobState> = states.into_iter().collect();

    for job in jobs.iter().filter(|j| j.server == server_id).cloned() {
        if job.state != JobState::Queued {
            continue;
        }
        if !job
            .dependencies()
            .iter()
            .all(|d| states_map.get(&d.job_id) == Some(&JobState::Succeeded))
        {
            continue;
        }

        let payload = build_generate_payload(&job, app);
        let mut running = job;
        if running.transition(JobState::Running, asset_studio_core::time::now()).is_err() {
            continue;
        }
        match client.submit(&payload, cancellation).await {
            Ok(_) => tracing::info!(job = %running.id.0, server = server_id, "job submitted"),
            Err(e) => {
                running.state = JobState::Failed;
                running.error = Some(format!("submit failed: {e}"));
                running.updated_at = asset_studio_core::time::now();
            }
        }
        let _ = app.library.lock().unwrap().update_job(&running);
    }
}

/// Build the `/generate` payload for a job, resolving its inputs to concrete
/// asset ids.
fn build_generate_payload(job: &Job, app: &Arc<App>) -> serde_json::Value {
    let mut inputs = serde_json::Map::new();
    {
        let lib = app.library.lock().unwrap();
        let lib: &Library = &lib;
        for (role, input) in &job.inputs {
            if let Ok(asset) = resolve_input(lib, input) {
                inputs.insert(role.clone(), serde_json::json!({ "asset": asset.id.0 }));
            }
        }
    }
    serde_json::json!({
        "task": job.task,
        "job_id": job.id.0,
        "prompt": job.prompt,
        "params": job.params,
        "count": job.count,
        "seeds": job.seeds,
        "inputs": serde_json::Value::Object(inputs),
    })
}

/// Resolve an [`InputRef`] to a concrete asset (a `JobOutput` resolves to the
/// dependency's `outputs[output_index]`). Takes an already-locked `&Library`.
fn resolve_input(lib: &Library, input: &InputRef) -> Result<Asset, asset_studio_library::StoreError> {
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
    lib.get_asset(&asset_id)?
        .ok_or(asset_studio_library::StoreError::Core(
            asset_studio_core::Error::UnknownAsset(asset_id.0.clone()),
        ))
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
        job.state = JobState::Failed;
        job.error = Some(reason.clone());
        job.updated_at = asset_studio_core::time::now();
        let _ = app.library.lock().unwrap().update_job(&job);
        tracing::warn!(job = %job.id.0, "job failed: {reason}");
    }
}
