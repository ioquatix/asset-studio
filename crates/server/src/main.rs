//! The `asset-studio-server` binary: boot the studio HTTP API on
//! `base-http-server` + `base-scheduler-native`.
//!
//! Usage:
//! ```sh
//! asset-studio-server --data /home/samuel/.asset-studio --listen 127.0.0.1:8090
//! ```
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use base_http_server::Options;
use base_scheduler::{
    Cancellation, Spawn, SocketBind, SocketListen, SocketLocalAddress, SocketOpen,
};
use base_scheduler_native::{NativeScheduler, NativeOptions};
use clap::Parser;
use futures_lite::future::block_on;

use asset_studio_launcher::Launcher;
use asset_studio_library::Library;
use asset_studio_server::{App, Studio, orchestrator};

#[derive(Debug, clap::Parser)]
#[command(name = "asset-studio-server", version)]
struct Cli {
    /// Data directory (SQLite library + blob store).
    #[arg(long, default_value = "~/.asset-studio")]
    data: PathBuf,

    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:8090")]
    listen: SocketAddr,

    /// Scheduler worker threads.
    #[arg(long, default_value_t = 2)]
    workers: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let cli = Cli::parse();

    let data = expand_tilde(&cli.data);
    std::fs::create_dir_all(&data)?;

    let library = Library::open(&data)?;
    let launcher = Launcher::new();
    let app = Arc::new(App::new(library, launcher));

    let scheduler = Arc::new(NativeScheduler::with_options(NativeOptions {
        workers: cli.workers,
        threads: cli.workers,
        capacity: 4096,
        timer_capacity: 1024,
    })?);

    let result = block_on(run(&scheduler, app, cli.listen));
    // Shut the scheduler down cleanly.
    block_on(scheduler.shutdown());
    result
}

async fn run(
    scheduler: &Arc<NativeScheduler>,
    app: Arc<App>,
    address: SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let handler = Studio::new(app.clone());
    let cancellation = Cancellation::never();

    let listener = scheduler
        .socket_open(address.into(), base_scheduler::SocketType::Stream, None, &cancellation)
        .await?;
    scheduler
        .socket_bind(&listener, address, &cancellation)
        .await?;
    scheduler
        .socket_listen(&listener, 128, &cancellation)
        .await?;

    let bound = scheduler.socket_local_address(&listener)?;
    eprintln!("asset-studio-server listening on http://{bound}");

    // One task per model server + the HTTP API, all spawned as `Send + 'static`
    // tasks on the scheduler (each captures an `Arc<NativeScheduler>` so the
    // task can use the scheduler's capabilities).
    let server_ids: Vec<String> =
        asset_studio_models::all_servers().iter().map(|s| s.id.clone()).collect();

    // Keep an Arc for `.spawn()` (the method takes `&self`), and clone into
    // each spawned future so the tasks can use the scheduler.
    let spawn_scheduler = Arc::clone(scheduler);
    let mut futures: Vec<NativeSchedulerHandle> = Vec::new();

    // The HTTP API.
    {
        let sch = Arc::clone(scheduler);
        let cancellation = cancellation.clone();
        let fut = async move {
            let result = base_http_server::serve(
                sch.as_ref(),
                listener,
                &handler,
                &Options::default(),
                &cancellation,
            )
            .await;
            if let Err(e) = result {
                tracing::error!("http server failed: {e}");
            }
        };
        futures.push(spawn_scheduler.spawn(fut)?);
    }

    // One orchestrator per model server.
    for id in server_ids {
        let sch = Arc::clone(scheduler);
        let app = Arc::clone(&app);
        let cancellation = cancellation.clone();
        let fut = async move {
            match orchestrator::run(sch.as_ref(), &app, &id, &cancellation).await {
                Ok(()) => {}
                Err(e) => tracing::warn!(server = %id, "orchestrator stopped: {e}"),
            }
        };
        futures.push(spawn_scheduler.spawn(fut)?);
    }

    // Run until all spawned tasks complete.
    join_all(futures).await;
    Ok(())
}

/// The handle to a spawned task.
type NativeSchedulerHandle = <NativeScheduler as base_scheduler::Spawn>::Handle<()>;

/// Await all spawned tasks until they complete.
async fn join_all(futures: Vec<NativeSchedulerHandle>) {
    for handle in futures {
        let _ = handle.await;
    }
}

fn expand_tilde(path: &std::path::Path) -> PathBuf {
    if let Ok(rest) = path.strip_prefix("~") {
        if let Some(home) = std::env::var_os("HOME") {
            return std::path::PathBuf::from(home).join(rest);
        }
    }
    path.to_path_buf()
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,asset_studio=debug"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init();
}
