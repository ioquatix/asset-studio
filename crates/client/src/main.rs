//! The `asset-studio` CLI: talk to `asset-studio-server` over HTTP.
//!
//! Built on `base-http-client` + `base-scheduler-native`. Each command opens a
//! pooled client to the studio, runs the command (zipped with the client
//! driver so calls progress), and prints JSON (or writes a file for `download`).
//!
//! Usage:
//! ```sh
//! asset-studio --studio 127.0.0.1:8090 servers
//! asset-studio import chicken.glb --kind mesh
//! asset-studio generate --server trellis2 --task 3d-paint \
//!     --input mesh=chicken.glb --input image=ref.png --wait
//! asset-studio wait <job-id>
//! ```
use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use base_scheduler::{Cancellation, Clock};
use base_scheduler_native::{NativeScheduler, NativeOptions};
use clap::Parser;
use futures_lite::future::{block_on, zip};

use asset_studio_client::StudioClient;

#[derive(Debug, Parser)]
#[command(name = "asset-studio", version, about = "asset-studio CLI")]
struct Cli {
    /// Studio server address.
    #[arg(long, global = true, default_value = "127.0.0.1:8090")]
    studio: SocketAddr,

    #[command(subcommand)]
    command: Command,
}

/// A `role=path` pair for `generate --input`.
#[derive(Debug, Clone)]
struct Input {
    role: String,
    path: PathBuf,
}

impl std::str::FromStr for Input {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (role, path) = s
            .split_once('=')
            .ok_or_else(|| format!("expected role=path, got {s}"))?;
        Ok(Self {
            role: role.to_string(),
            path: PathBuf::from(path),
        })
    }
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    /// Show the model servers + their tasks.
    Servers,
    /// List resources (assets or jobs).
    List {
        /// What to list: `assets` or `jobs`.
        #[arg(long, default_value = "assets")]
        what: String,
        /// For jobs: filter by state (e.g. `queued`, `succeeded`).
        #[arg(long)]
        state: Option<String>,
    },
    /// Import a file as an asset.
    Import {
        /// File to import.
        file: PathBuf,
        /// Asset kind: `image` or `mesh`.
        #[arg(long, default_value = "image")]
        kind: String,
        /// Asset title.
        #[arg(long)]
        title: Option<String>,
    },
    /// Submit a generation job.
    Generate {
        /// Model server id (e.g. `trellis2`, `hunyuan3d`).
        #[arg(long)]
        server: String,
        /// Task name (e.g. `3d-paint`).
        #[arg(long)]
        task: String,
        /// Inputs as `role=path` (each file is imported first).
        #[arg(long)]
        input: Vec<Input>,
        /// Text prompt.
        #[arg(long)]
        prompt: Option<String>,
        /// Task params as a JSON object string.
        #[arg(long)]
        params: Option<String>,
        /// Number of outputs.
        #[arg(long, default_value_t = 1)]
        count: u64,
        /// Idempotency key (dedup on resubmit).
        #[arg(long)]
        idempotency_key: Option<String>,
        /// Poll until the job reaches a terminal state.
        #[arg(long)]
        wait: bool,
    },
    /// Poll a job until it reaches a terminal state.
    Wait {
        /// Job id.
        id: String,
        /// Poll interval in milliseconds.
        #[arg(long, default_value_t = 500)]
        interval_ms: u64,
    },
    /// Download an asset's bytes to a file (or stdout if no `-o`).
    Download {
        /// Asset id.
        id: String,
        /// Output file (defaults to stdout).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let scheduler = NativeScheduler::with_options(NativeOptions {
        workers: 2,
        threads: 2,
        capacity: 1024,
        timer_capacity: 256,
    })?;
    let result = block_on(run(&scheduler, cli.studio, cli.command));
    block_on(scheduler.shutdown());
    result
}

async fn run(scheduler: &NativeScheduler, studio: SocketAddr, command: Command) -> Result<()> {
    let (client, driver) = StudioClient::connect(scheduler, studio)?;
    // The driver runs on its own lifetime signal; cancel it once the command
    // finishes so `driver.run` can retire its connections and the zip returns.
    let driver_lifetime = Cancellation::new();
    let cancellation = Cancellation::never();
    let output = zip(
        async {
            let result = execute(scheduler, &client, command, &cancellation).await;
            driver_lifetime.cancel();
            result
        },
        driver.run(&driver_lifetime),
    )
    .await
    .0;
    if let Ok(text) = &output {
        if !text.is_empty() {
            println!("{text}");
        }
    }
    output?;
    Ok(())
}

/// Dispatch a [`Command`] to the matching API call, returning the text to print.
async fn execute(
    scheduler: &NativeScheduler,
    client: &StudioClient<'_>,
    command: Command,
    cancellation: &Cancellation,
) -> Result<String> {
    match command {
        Command::Servers => Ok(json_value(client.servers(cancellation).await?)),
        Command::List { what, state } => match what.as_str() {
            "jobs" => Ok(json_value(client.jobs(state.as_deref(), cancellation).await?)),
            "assets" | _ => Ok(json_value(
                client.assets(state.as_deref(), cancellation).await?,
            )),
        },
        Command::Import { file, kind, title } => {
            let bytes = std::fs::read(&file).map_err(anyhow::Error::from)?;
            Ok(json_value(client.import(&kind, title.as_deref(), &bytes, cancellation).await?))
        }
        Command::Generate {
            server,
            task,
            input,
            prompt,
            params,
            count,
            idempotency_key,
            wait,
        } => {
            // Import each input file, remembering role -> asset id.
            let mut inputs = serde_json::Map::new();
            for input in &input {
                let bytes = std::fs::read(&input.path).map_err(anyhow::Error::from)?;
                let kind = kind_for_role(&input.role);
                let title = input
                    .path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string());
                let asset = client
                    .import(&kind, title.as_deref(), &bytes, cancellation)
                    .await?;
                let id = asset["asset"]["id"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("import response had no asset id"))?
                    .to_string();
                inputs.insert(
                    input.role.clone(),
                    serde_json::json!({ "asset": id }),
                );
            }

            let payload = serde_json::json!({
                "server": server,
                "task": task,
                "prompt": prompt,
                "params": params
                    .as_deref()
                    .map(serde_json::from_str::<serde_json::Value>)
                    .transpose()
                    .map_err(anyhow::Error::from)?
                    .unwrap_or(serde_json::Value::Null),
                "count": count,
                "inputs": serde_json::Value::Object(inputs),
                "idempotency_key": idempotency_key,
            });

            let mut job = client.submit(&payload, cancellation).await?;
            if wait {
                let final_job = wait_for_job(scheduler, client, &job, 500, cancellation).await?;
                job = serde_json::json!({ "job": final_job });
            }
            Ok(json_value(job))
        }
        Command::Wait { id, interval_ms } => {
            let job = client.job(&id, cancellation).await?;
            let final_job = wait_for_job(scheduler, client, &job, interval_ms, cancellation).await?;
            Ok(json_value(serde_json::json!({ "job": final_job })))
        }
        Command::Download { id, output } => {
            let bytes = client.download(&id, cancellation).await?;
            match output {
                Some(path) => {
                    std::fs::write(&path, &bytes).map_err(anyhow::Error::from)?;
                    Ok(format!("wrote {} bytes to {}", bytes.len(), path.display()))
                }
                None => {
                    std::io::stdout().write_all(&bytes).map_err(anyhow::Error::from)?;
                    Ok(String::new())
                }
            }
        }
    }
}

/// Poll a job until it reaches a terminal state, returning the final job.
///
/// `job` may be the full `{"job": {...}}` API envelope or the bare job object;
/// this unwraps the `job` field if present.
async fn wait_for_job(
    scheduler: &NativeScheduler,
    client: &StudioClient<'_>,
    job: &serde_json::Value,
    interval_ms: u64,
    cancellation: &Cancellation,
) -> Result<serde_json::Value> {
    let mut job = job.get("job").cloned().unwrap_or_else(|| job.clone());
    let id = job["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("job had no id"))?
        .to_string();
    while !is_terminal_state(job.get("state").and_then(|s| s.as_str())) {
        let _ = scheduler
            .sleep(std::time::Duration::from_millis(interval_ms), cancellation)
            .await;
        let response = client.job(&id, cancellation).await?;
        job = response.get("job").cloned().unwrap_or(response);
        if let Some(state) = job.get("state").and_then(|s| s.as_str()) {
            eprintln!("job {id}: {state}");
        }
    }
    Ok(job)
}

fn is_terminal_state(state: Option<&str>) -> bool {
    matches!(state, Some("succeeded" | "failed" | "cancelled"))
}

/// Map an input role to the asset kind to import it as.
fn kind_for_role(role: &str) -> String {
    if role.contains("mesh") || role.contains("glb") || role.contains("model") {
        "mesh".to_string()
    } else {
        "image".to_string()
    }
}

fn json_value(value: serde_json::Value) -> String {
    serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_parses_role_equals_path() {
        let input: Input = "mesh=chicken.glb".parse().unwrap();
        assert_eq!(input.role, "mesh");
        assert_eq!(input.path, PathBuf::from("chicken.glb"));
    }

    #[test]
    fn input_rejects_missing_separator() {
        assert!("chicken.glb".parse::<Input>().is_err());
    }

    #[test]
    fn kind_for_role_maps_mesh_roles() {
        assert_eq!(kind_for_role("mesh"), "mesh");
        assert_eq!(kind_for_role("image"), "image");
        assert_eq!(kind_for_role("glb_model"), "mesh");
    }

    #[test]
    fn terminal_states_are_recognized() {
        assert!(is_terminal_state(Some("succeeded")));
        assert!(is_terminal_state(Some("failed")));
        assert!(is_terminal_state(Some("cancelled")));
        assert!(!is_terminal_state(Some("running")));
        assert!(!is_terminal_state(Some("queued")));
        assert!(!is_terminal_state(None));
    }
}
