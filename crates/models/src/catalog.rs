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
    /// Free GPU memory (MiB) it needs to start; the orchestrator waits for it.
    pub vram_mib: u64,
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
/// The root is `ASSET_STUDIO_ROOT` (default
/// `~/Developer/kvladsrc/gym/assets_studio`), the checked-out reference repo. Returns `None` if the home dir is unknown (so a
/// launch without an override degrades to the inherited cwd).
fn server_dir(subdir: &str) -> Option<PathBuf> {
    let root = env::var("ASSET_STUDIO_ROOT").ok().map(PathBuf::from).or_else(|| {
        env::var_os("HOME")
            .map(|h| PathBuf::from(h).join("Developer/kvladsrc/gym/assets_studio"))
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

/// The asset-studio checkout (for `scripts/` and `model_servers/`): the
/// compile-time location, unless `ASSET_STUDIO_HOME` says otherwise.
fn studio_home() -> PathBuf {
    env::var_os("ASSET_STUDIO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")))
}

/// A GPU model server launched by `scripts/model_server.sh` in the Nix CUDA
/// shell: `project` is its uv project, `venv` its environment under
/// `~/.cache/assets-studio/venvs/`, `script` its console script.
#[allow(clippy::too_many_arguments)]
fn gpu_server(
    id: &str,
    title: &str,
    port: u16,
    vram_mib: u64,
    project: Option<PathBuf>,
    venv: &str,
    script: &str,
    tasks: Vec<TaskInfo>,
) -> ModelServer {
    let launcher = studio_home().join("scripts/model_server.sh");
    let project = project.unwrap_or_default();
    ModelServer {
        id: id.into(),
        title: title.into(),
        port,
        uses_gpu: true,
        vram_mib,
        program: "/usr/bin/bash".into(),
        args: vec![
            launcher.to_string_lossy().into_owned(),
            project.to_string_lossy().into_owned(),
            venv.into(),
            script.into(),
        ],
        cwd: None,
        env: vec![("TORCH_CUDA_ARCH_LIST".into(), "8.9".into()), port_env(port)],
        tasks,
    }
}

/// `server` with one more environment variable for its process.
fn with_env(mut server: ModelServer, key: &str, value: &str) -> ModelServer {
    server.env.push((key.into(), value.into()));
    server
}

/// The full catalog of known servers. Built on demand (cheap: a few entries).
///
/// `zimage` → `birefnet` → `trellis2_3d` is the text-to-3D chain: a prompt
/// becomes a picture, the picture a cut-out, the cut-out a textured GLB.
/// `trellis2` paints an existing mesh from a cut-out.
pub fn all_servers() -> Vec<ModelServer> {
    let own = Some(studio_home().join("model_servers"));
    vec![
        gpu_server(
            "zimage",
            "Z-Image-Turbo",
            8127,
            14_000,
            own.clone(),
            "studio_servers",
            "asset-studio-zimage-server",
            vec![task(
                "text-to-image",
                "A picture from a prompt (params.object_reference frames one object for image-to-3D).",
                &[],
                true,
            )],
        ),
        gpu_server(
            "birefnet",
            "BiRefNet",
            8128,
            3_000,
            server_dir("birefnet"),
            "birefnet",
            "birefnet-model-server",
            vec![task(
                "image-to-image",
                "Remove the background: the same picture as a cut-out RGBA PNG.",
                &[("image", "image/png"), ("image", "image/jpeg"), ("image", "image/webp")],
                false,
            )],
        ),
        with_env(gpu_server(
            "trellis2_3d",
            "TRELLIS.2 Image to 3D",
            8129,
            // All models stay on the GPU (TRELLIS2_LOW_VRAM=0): ~15 GiB of
            // weights plus activations on a 24 GB card, and only a few GB of
            // host RAM. Low-VRAM mode needs 16-31 GB of host RAM instead,
            // which was killed by the kernel's OOM killer on a 46 GB machine
            // with a full /tmp; set ASSET_STUDIO_TRELLIS2_3D_LOW_VRAM=1 for
            // cards under 24 GB.
            20_000,
            own,
            "studio_servers",
            "asset-studio-trellis2-server",
            vec![task(
                "image-to-3d",
                "A textured PBR GLB from a cut-out picture (RGBA PNG).",
                &[("image", "image/png")],
                false,
            )],
        ), "TRELLIS2_LOW_VRAM", &env::var("ASSET_STUDIO_TRELLIS2_3D_LOW_VRAM").unwrap_or_else(|_| "0".into())),
        gpu_server(
            "trellis2",
            "TRELLIS.2 Paint",
            8123,
            6_000,
            server_dir("trellis2_paint"),
            "trellis2_paint",
            "trellis2-paint-model-server",
            vec![task(
                "3d-paint",
                "Paint an existing mesh with a reference image, preserving surface and UVs.",
                &[("mesh", "model/gltf-binary"), ("image", "image/png")],
                false,
            )],
        ),
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
    fn spec_launches_through_the_wrapper_with_a_pinned_port() {
        let s = find_server("trellis2_3d").unwrap();
        assert_eq!(s.url(), "http://127.0.0.1:8129");
        let spec = s.spec();
        assert_eq!(spec.id, "trellis2_3d");
        assert!(spec.uses_gpu);
        assert_eq!(spec.program, "/usr/bin/bash");
        assert!(spec.args[0].ends_with("scripts/model_server.sh"));
        assert!(spec.args[1].ends_with("model_servers"));
        assert_eq!(&spec.args[2..], ["studio_servers", "asset-studio-trellis2-server"]);
        // The launch pins the port the client will dial.
        assert!(spec.env.iter().any(|(k, v)| k == "MODEL_SERVER_PORT" && v == "8129"));
        assert_eq!(s.info().tasks.len(), s.tasks.len());
    }

    #[test]
    fn the_text_to_3d_chain_is_in_the_catalog() {
        let ports: std::collections::BTreeSet<u16> = all_servers().iter().map(|s| s.port).collect();
        assert_eq!(ports.len(), all_servers().len(), "ports are unique");
        assert!(find_server("zimage").unwrap().task("text-to-image").unwrap().uses_prompt);
        assert!(find_server("birefnet").unwrap().task("image-to-image").is_some());
        assert!(find_server("trellis2_3d").unwrap().task("image-to-3d").is_some());
    }
}
