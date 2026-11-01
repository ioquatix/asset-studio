//! The catalog of known model servers.
//!
//! Each entry describes *what* a server offers (its tasks and input roles) and
//! *how* to launch it (a [`ServerSpec`] for the launcher). The servers are the
//! reference `assets_studio/model_servers/*` Python packages, each a `uv`
//! project exposing a console script (`trellis2-paint-model-server`, …).
//!
//! The launch is `uv run --no-sync <script>` in the server's project directory
//! (so `uv` can resolve the venv), with `MODEL_SERVER_PORT` pinned so the studio
//! and the server agree on the port. Everything — program, args, working
//! directory, and the per-server env — is overridable via
//! `ASSET_STUDIO_<ID>_CMD` / `ASSET_STUDIO_<ID>_ARGS` / `ASSET_STUDIO_<ID>_CWD`
//! / `ASSET_STUDIO_<ID>_PORT` so the entrypoints can be re-pointed without
//! recompiling (and so the GPU devshell's `just`-style invocation can be
//! substituted in).
use asset_studio_core::TaskInfo;
use asset_studio_launcher::ServerSpec;
use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

/// A known model server and how to run it.
#[derive(Debug, Clone)]
pub struct ModelServer {
    /// Stable id (e.g. `trellis2`).
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// The TCP port this server listens on (pinned via `MODEL_SERVER_PORT`).
    pub port: u16,
    /// Whether it needs the GPU slot.
    pub uses_gpu: bool,
    /// Default program to run (e.g. `uv`).
    pub program: String,
    /// Default arguments (e.g. `run --no-sync <script>`).
    pub args: Vec<String>,
    /// Working directory (the server's `pyproject` dir).
    pub cwd: Option<PathBuf>,
    /// Extra environment for the child.
    pub env: Vec<(String, String)>,
    /// The tasks this server offers.
    pub tasks: Vec<TaskInfo>,
}

impl ModelServer {
    /// The base URL of a running instance of this server.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// The core [`ServerInfo`](asset_studio_core::ServerInfo) for this server.
    pub fn info(&self) -> asset_studio_core::ServerInfo {
        asset_studio_core::ServerInfo {
            id: self.id.clone(),
            title: self.title.clone(),
            url: self.url(),
            tasks: self.tasks.clone(),
        }
    }

    /// The launcher [`ServerSpec`] for this server, honoring env overrides.
    pub fn spec(&self) -> ServerSpec {
        let id_up = self.id.to_uppercase();
        let (program, args) = env_overridden(
            &self.id,
            &self.program,
            &self.args,
            &format!("ASSET_STUDIO_{id_up}_CMD"),
            &format!("ASSET_STUDIO_{id_up}_ARGS"),
        );
        let cwd = env_path(&format!("ASSET_STUDIO_{id_up}_CWD")).or_else(|| self.cwd.clone());
        let mut env = self.env.clone();
        if let Ok(port) = env::var(&format!("ASSET_STUDIO_{id_up}_PORT")) {
            env.retain(|(k, _)| k != "MODEL_SERVER_PORT");
            env.push(("MODEL_SERVER_PORT".into(), port));
        }
        ServerSpec {
            id: self.id.clone(),
            program,
            args,
            cwd,
            env,
            uses_gpu: self.uses_gpu,
        }
    }

    /// A task by name.
    pub fn task(&self, name: &str) -> Option<&TaskInfo> {
        self.tasks.iter().find(|t| t.name == name)
    }
}

/// A `MODEL_SERVER_PORT`-tagged env pair for a server's launch.
fn port_env(port: u16) -> (String, String) {
    ("MODEL_SERVER_PORT".into(), port.to_string())
}

/// The model server's project directory under the asset-studio source root.
///
/// The root is `ASSET_STUDIO_ROOT` (default `~/gym/assets_studio`), matching the
/// checked-out reference repo. Returns `None` if the home dir is unknown (so a
/// launch without an override degrades to the inherited cwd).
fn server_dir(subdir: &str) -> Option<PathBuf> {
    let root = env::var("ASSET_STUDIO_ROOT").ok().map(PathBuf::from).or_else(|| {
        env::var_os("HOME")
            .map(|h| PathBuf::from(h).join("gym").join("assets_studio"))
    })?;
    Some(root.join("model_servers").join(subdir))
}

/// `ASSET_STUDIO_<ID>_CMD` / `_ARGS` override the program + arg list.
fn env_overridden(
    _id: &str,
    program: &str,
    args: &[String],
    key_cmd: &str,
    key_args: &str,
) -> (String, Vec<String>) {
    if let Ok(cmd) = env::var(key_cmd) {
        let arg_list: Vec<String> = match env::var(key_args) {
            Ok(a) => a.split_whitespace().map(String::from).collect(),
            Err(_) => vec![],
        };
        return (cmd, arg_list);
    }
    (program.to_string(), args.to_vec())
}

/// An optional `KEY=<path>` environment override.
fn env_path(key: &str) -> Option<PathBuf> {
    env::var(key).ok().map(PathBuf::from)
}

/// Build a task descriptor.
fn task(name: &str, description: &str, inputs: &[(&str, &str)], uses_prompt: bool) -> TaskInfo {
    let mut roles: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (role, mime) in inputs {
        roles.entry((*role).to_string()).or_default().push(mime.to_string());
    }
    TaskInfo {
        name: name.to_string(),
        description: description.to_string(),
        input_roles: roles,
        uses_prompt,
    }
}

/// The full catalog of known servers. Built on demand (cheap: ~4 entries).
pub fn all_servers() -> Vec<ModelServer> {
    vec![
        ModelServer {
            id: "trellis2".into(),
            title: "TRELLIS.2 Paint".into(),
            port: 8123,
            uses_gpu: true,
            program: "uv".into(),
            args: vec!["run".into(), "--no-sync".into(), "trellis2-paint-model-server".into()],
            cwd: server_dir("trellis2_paint"),
            env: vec![("TORCH_CUDA_ARCH_LIST".into(), "8.9".into()), port_env(8123)],
            tasks: vec![task(
                "3d-paint",
                "Paint an existing mesh with a reference image, preserving surface and UVs.",
                &[("mesh", "model/gltf-binary"), ("image", "image/png")],
                false,
            )],
        },
        ModelServer {
            id: "hunyuan3d".into(),
            title: "Hunyuan3D 2.1".into(),
            port: 8124,
            uses_gpu: true,
            program: "uv".into(),
            args: vec!["run".into(), "--no-sync".into(), "hunyuan3d21-model-server".into()],
            cwd: server_dir("hunyuan3d21"),
            env: vec![("TORCH_CUDA_ARCH_LIST".into(), "8.9".into()), port_env(8124)],
            tasks: vec![
                task(
                    "3d-shape",
                    "Generate a 3D shape from an image.",
                    &[("image", "image/png")],
                    false,
                ),
                task(
                    "3d-paint",
                    "Paint a generated shape with PBR textures.",
                    &[("mesh", "model/gltf-binary"), ("image", "image/png")],
                    false,
                ),
            ],
        },
        ModelServer {
            id: "hi3d".into(),
            title: "Hi3D".into(),
            port: 8125,
            uses_gpu: true,
            program: "uv".into(),
            args: vec!["run".into(), "--no-sync".into(), "hi3d-model-server".into()],
            cwd: server_dir("hi3d"),
            env: vec![("TORCH_CUDA_ARCH_LIST".into(), "8.9".into()), port_env(8125)],
            tasks: vec![task(
                "3d-image",
                "Generate a 3D mesh from an image.",
                &[("image", "image/png")],
                false,
            )],
        },
        ModelServer {
            id: "animation".into(),
            title: "Animation Tools".into(),
            port: 8126,
            uses_gpu: true,
            program: "uv".into(),
            args: vec!["run".into(), "--no-sync".into(), "animation-tools-model-server".into()],
            cwd: server_dir("animation_tools"),
            env: vec![port_env(8126)],
            tasks: vec![task(
                "motion",
                "Estimate a skeleton animation (SMPL-X / SOMA) from a video.",
                &[("video", "video/mp4")],
                false,
            )],
        },
    ]
}

/// Find a server by id.
pub fn find_server(id: &str) -> Option<ModelServer> {
    all_servers().into_iter().find(|s| s.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_nonempty_and_lookup_works() {
        let servers = all_servers();
        assert!(!servers.is_empty());
        let trellis = find_server("trellis2").expect("trellis2 exists");
        assert!(trellis.uses_gpu);
        assert_eq!(trellis.port, 8123);
        assert!(trellis.task("3d-paint").is_some());
        assert!(find_server("does-not-exist").is_none());
    }

    #[test]
    fn spec_uses_uv_and_pinned_port() {
        let s = find_server("trellis2").unwrap();
        assert_eq!(s.url(), "http://127.0.0.1:8123");
        let spec = s.spec();
        assert_eq!(spec.id, "trellis2");
        assert!(spec.uses_gpu);
        assert_eq!(spec.program, "uv");
        assert!(spec.args.starts_with(&["run".into(), "--no-sync".into()]));
        // The launch pins the port the client will dial.
        assert!(
            spec.env.iter().any(|(k, v)| k == "MODEL_SERVER_PORT" && v == "8123")
        );
        // Info carries the tasks.
        let info = s.info();
        assert_eq!(info.tasks.len(), s.tasks.len());
    }
}
