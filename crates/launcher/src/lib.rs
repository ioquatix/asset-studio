//! Model-server process lifecycle for asset-studio.
//!
//! The [`Launcher`] owns the *process* side of running a model server:
//!
//! * **Spawn** — start a server's program (a Python model server) with its
//!   environment, via `std::process::Command`.
//! * **GPU slot** — a capacity-1 admission gate so at most one GPU-hungry
//!   server runs at a time (one 24 GB card). A server whose `uses_gpu` is set
//!   acquires the slot before spawning and the slot is released only when *that
//!   server* stops (tracked per-server, so a non-GPU server exiting doesn't
//!   free it).
//! * **Reap** — `poll` (non-blocking `try_wait`), `kill`, and `process_id`
//!   (for the orchestrator to adopt the child into the scheduler's
//!   [`ProcessWait`](base_scheduler::ProcessWait) and await termination
//!   asynchronously).
//!
//! Readiness (HTTP `GET /health`) is the *orchestrator's* concern, not the
//! launcher's; the launcher only knows the process state.
use std::collections::{HashMap, HashSet};
use std::io;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;

/// A declared command to launch a model server.
#[derive(Debug, Clone)]
pub struct ServerSpec {
    /// Stable server id (e.g. `trellis2`).
    pub id: String,
    /// The program to run (e.g. `uv` or `python`).
    pub program: String,
    /// Arguments to the program.
    pub args: Vec<String>,
    /// Working directory for the child (e.g. the server's `pyproject` dir, so
    /// `uv` can resolve the project). `None` inherits the studio's cwd.
    pub cwd: Option<PathBuf>,
    /// Extra environment variables.
    pub env: Vec<(String, String)>,
    /// Whether this server needs the GPU slot.
    pub uses_gpu: bool,
}

/// The observed state of a managed server process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerStatus {
    /// Not started.
    Off,
    /// Spawned and believed alive (not yet confirmed exited).
    Running,
    /// Exited with a status code.
    Exited(i32),
}

struct ServerHandle {
    process: Child,
    uses_gpu: bool,
}

/// Manages the set of in-flight model-server processes plus the GPU slot.
///
/// Thread-safe: all state is behind a single mutex (process operations are
/// short-critical; the GPU slot never blocks on I/O here).
pub struct Launcher {
    state: Mutex<LauncherState>,
}

struct LauncherState {
    servers: HashMap<String, ServerHandle>,
    /// The ids of servers currently holding the GPU slot (capacity 1).
    gpu_holders: HashSet<String>,
}

impl Default for Launcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Launcher {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(LauncherState {
                servers: HashMap::new(),
                gpu_holders: HashSet::new(),
            }),
        }
    }

    /// Whether the GPU slot is currently free.
    pub fn gpu_is_free(&self) -> bool {
        self.state.lock().unwrap().gpu_holders.is_empty()
    }

    /// The ids of servers currently holding the GPU slot.
    pub fn gpu_holders(&self) -> Vec<String> {
        self.state.lock().unwrap().gpu_holders.iter().cloned().collect()
    }

    /// The set of server ids currently tracked.
    pub fn server_ids(&self) -> Vec<String> {
        self.state.lock().unwrap().servers.keys().cloned().collect()
    }

    /// Start `spec`: acquire the GPU slot if needed (returning `Err` if it's
    /// already held by another server), then spawn the process. Returns the
    /// platform process id.
    pub fn spawn(&self, spec: &ServerSpec) -> io::Result<u32> {
        let mut state = self.state.lock().unwrap();
        if spec.uses_gpu && !state.gpu_holders.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "GPU slot is in use by another model server",
            ));
        }

        let mut cmd = Command::new(&spec.program);
        cmd.args(&spec.args);
        if let Some(cwd) = &spec.cwd {
            cmd.current_dir(cwd);
        }
        for (k, v) in &spec.env {
            cmd.env(k, v);
        }
        // Detach stdio so a long-running server never blocks on a full pipe.
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        if spec.uses_gpu {
            state.gpu_holders.insert(spec.id.clone());
        }
        let pid = child.id();
        state.servers.insert(
            spec.id.clone(),
            ServerHandle { process: child, uses_gpu: spec.uses_gpu },
        );
        Ok(pid)
    }

    /// Non-blockingly reap the process if it has exited (and free the GPU slot
    /// for it if it held one). Returns the current status.
    pub fn poll(&self, id: &str) -> io::Result<ServerStatus> {
        let mut state = self.state.lock().unwrap();
        let handle = state
            .servers
            .get_mut(id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, id))?;
        match handle.process.try_wait()? {
            Some(status) => {
                let code = status.code().unwrap_or(-1);
                if handle.uses_gpu {
                    state.gpu_holders.remove(id);
                }
                state.servers.remove(id);
                Ok(ServerStatus::Exited(code))
            }
            None => Ok(ServerStatus::Running),
        }
    }

    /// The platform process id of a running server, for adopting into
    /// [`ProcessWait`](base_scheduler::ProcessWait).
    pub fn process_id(&self, id: &str) -> io::Result<u32> {
        let state = self.state.lock().unwrap();
        let handle = state
            .servers
            .get(id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, id))?;
        Ok(handle.process.id())
    }

    /// Force-stop a server (SIGKILL), reap it, and free the GPU slot if it
    /// held one. Returns the exit code, if it was tracked.
    pub fn kill(&self, id: &str) -> io::Result<Option<i32>> {
        let mut state = self.state.lock().unwrap();
        let handle = match state.servers.remove(id) {
            Some(h) => h,
            None => return Ok(None),
        };
        let release = handle.uses_gpu;
        let mut process = handle.process;
        let _ = process.kill();
        let status = process.wait()?;
        if release {
            state.gpu_holders.remove(id);
        }
        Ok(Some(status.code().unwrap_or(-1)))
    }

    /// Stop every tracked server.
    pub fn stop_all(&self) {
        let ids: Vec<String> = self.server_ids();
        for id in ids {
            let _ = self.kill(&id);
        }
    }
}

impl Drop for Launcher {
    fn drop(&mut self) {
        self.stop_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str, uses_gpu: bool, program: &str, args: &[&str]) -> ServerSpec {
        ServerSpec {
            id: id.into(),
            program: program.into(),
            args: args.iter().map(|s| s.to_string()).collect(),
            cwd: None,
            env: vec![],
            uses_gpu,
        }
    }

    #[test]
    fn gpu_slot_capacity_one() {
        let l = Launcher::new();
        // Spin a GPU "server" (sleep) and confirm the slot is taken.
        l.spawn(&spec("g1", true, "sleep", &["5"])).unwrap();
        assert!(!l.gpu_is_free());
        // A second GPU server must be rejected.
        assert!(l.spawn(&spec("g2", true, "sleep", &["5"])).is_err());
        // A non-GPU server is unaffected.
        l.spawn(&spec("c1", false, "sleep", &["5"])).unwrap();
        assert_eq!(l.gpu_holders(), vec!["g1".to_string()]);
        l.stop_all();
        assert!(l.gpu_is_free());
    }

    #[test]
    fn poll_reports_exit_and_frees_gpu() {
        let l = Launcher::new();
        l.spawn(&spec("g1", true, "true", &[])).unwrap();
        assert!(!l.gpu_is_free());
        // `true` exits immediately; poll should observe the exit and free the slot
        // (give it a moment via repeated non-blocking polls).
        use std::time::Duration;
        for _ in 0..50 {
            match l.poll("g1").unwrap() {
                ServerStatus::Exited(_) => break,
                _ => std::thread::sleep(Duration::from_millis(20)),
            }
        }
        assert!(l.gpu_is_free());
        assert!(l.server_ids().is_empty());
    }

    #[test]
    fn process_id_is_positive() {
        let l = Launcher::new();
        l.spawn(&spec("c1", false, "sleep", &["3"])).unwrap();
        assert!(l.process_id("c1").unwrap() > 0);
        l.kill("c1").unwrap();
        assert_eq!(l.kill("c1").unwrap(), None);
    }
}
