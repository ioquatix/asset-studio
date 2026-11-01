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
    Cancellation, SocketBind, SocketListen, SocketLocalAddress, SocketOpen,
};
use base_scheduler_native::{NativeScheduler, NativeOptions};
use clap::Parser;
use futures_lite::future::block_on;

use asset_studio_launcher::Launcher;
use asset_studio_library::Library;
use asset_studio_server::{App, Studio};

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

    let scheduler = NativeScheduler::with_options(NativeOptions {
        workers: cli.workers,
        threads: cli.workers,
        capacity: 4096,
        timer_capacity: 1024,
    })?;

    let result = block_on(run(&scheduler, app, cli.listen));
    // Shut the scheduler down cleanly.
    block_on(scheduler.shutdown());
    result
}

async fn run(
    scheduler: &NativeScheduler,
    app: Arc<App>,
    address: SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let handler = Studio::new(app);
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

    base_http_server::serve(
        scheduler,
        listener,
        &handler,
        &Options::default(),
        &cancellation,
    )
    .await?;
    Ok(())
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
