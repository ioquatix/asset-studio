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
//! Readiness (`GET /v1/info`) is the *orchestrator's* concern, not the
//! launcher's; the launcher only knows the process state (and, through
//! [`gpu_free_mib`], how much GPU memory is free before a start).
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

    /// Stop a server and every process it started (the launch wrapper, `uv`,
    /// the Python server): `SIGTERM` to the whole tree, up to 10 s for the
    /// server to shut down cleanly, then `SIGKILL` to whatever remains. Reaps
    /// it and frees the GPU slot if it held one. Returns the exit code, if it
    /// was tracked.
    pub fn kill(&self, id: &str) -> io::Result<Option<i32>> {
        let mut state = self.state.lock().unwrap();
        let handle = match state.servers.remove(id) {
            Some(h) => h,
            None => return Ok(None),
        };
        let release = handle.uses_gpu;
        let mut process = handle.process;
        let tree = process_tree(process.id());
        signal(&tree, "TERM");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut status = None;
        while std::time::Instant::now() < deadline {
            if let Some(s) = process.try_wait()? {
                status = Some(s);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        // Descendants that outlived their parent (or ignored SIGTERM).
        let survivors: Vec<u32> = tree.into_iter().filter(|pid| alive(*pid)).collect();
        signal(&survivors, "KILL");
        let status = match status {
            Some(s) => s,
            None => {
                let _ = process.kill();
                process.wait()?
            }
        };
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

/// `root` and all its descendants (from `/proc/<pid>/task/*/children`), root first.
fn process_tree(root: u32) -> Vec<u32> {
    let mut tree = vec![root];
    let mut index = 0;
    while index < tree.len() {
        let pid = tree[index];
        index += 1;
        let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
            continue;
        };
        for task in tasks.flatten() {
            if let Ok(children) = std::fs::read_to_string(task.path().join("children")) {
                tree.extend(children.split_whitespace().filter_map(|c| c.parse::<u32>().ok()));
            }
        }
    }
    tree
}

fn alive(pid: u32) -> bool {
    // A zombie still has a /proc entry; only count processes that can run.
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .map(|stat| !stat.rsplit(')').next().unwrap_or("").trim_start().starts_with('Z'))
        .unwrap_or(false)
}

fn signal(pids: &[u32], name: &str) {
    if pids.is_empty() {
        return;
    }
    let _ = Command::new("kill")
        .arg(format!("-{name}"))
        .args(pids.iter().map(u32::to_string))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

/// Free memory on the first GPU, in MiB, from `nvidia-smi`; `None` if it
/// cannot be queried (no NVIDIA driver, or `nvidia-smi` missing), in which
/// case callers should not block on it.
pub fn gpu_free_mib() -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_free_mib(&String::from_utf8_lossy(&output.stdout))
}

/// The first line of `nvidia-smi --query-gpu=memory.free` output, in MiB.
fn parse_free_mib(text: &str) -> Option<u64> {
    text.lines().next()?.trim().parse().ok()
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
    fn kill_stops_grandchildren_too() {
        let l = Launcher::new();
        // A shell that starts a grandchild, as the launch wrapper does.
        l.spawn(&spec("tree", false, "sh", &["-c", "sleep 30 & wait"])).unwrap();
        let root = l.process_id("tree").unwrap();
        let mut tree = vec![];
        for _ in 0..50 {
            tree = process_tree(root);
            if tree.len() > 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(tree.len() > 1, "the grandchild started");
        l.kill("tree").unwrap();
        assert!(tree.iter().all(|pid| !alive(*pid)), "no process of the tree survives");
    }

    #[test]
    fn parses_nvidia_smi_free_memory() {
        assert_eq!(parse_free_mib("22625\n"), Some(22625));
        assert_eq!(parse_free_mib("  812 \n2048\n"), Some(812));
        assert_eq!(parse_free_mib("[N/A]\n"), None);
        assert_eq!(parse_free_mib(""), None);
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
