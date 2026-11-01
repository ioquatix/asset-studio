//! The catalog of known model servers.
//!
//! Each entry describes *what* a server offers (its tasks and input roles) and
//! *how* to launch it (a [`ServerSpec`] for the launcher). The program and
//! module are the one thing that must match the reference `model_servers/`;
//! they default to `python -m <module>` and are overridable via the
//! `ASSET_STUDIO_<ID>_CMD` / `ASSET_STUDIO_<ID>_ARGS` environment variables so
//! the entrypoints can be re-pointed without recompiling.
use asset_studio_core::TaskInfo;
use asset_studio_launcher::ServerSpec;
use std::collections::BTreeMap;
use std::env;

/// A known model server and how to run it.
#[derive(Debug, Clone)]
pub struct ModelServer {
    /// Stable id (e.g. `trellis2`).
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// The TCP port this server listens on.
    pub port: u16,
    /// Whether it needs the GPU slot.
    pub uses_gpu: bool,
    /// The program to run (default `python`).
    pub program: String,
    /// Arguments to the program (default `-m <module>`).
    pub args: Vec<String>,
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
        let (program, args) = env_override(&self.id, &self.program, &self.args);
        ServerSpec {
            id: self.id.clone(),
            program,
            args,
            env: self.env.clone(),
            uses_gpu: self.uses_gpu,
        }
    }

    /// A task by name.
    pub fn task(&self, name: &str) -> Option<&TaskInfo> {
        self.tasks.iter().find(|t| t.name == name)
    }
}

fn env_override(
    id: &str,
    program: &str,
    args: &[String],
) -> (String, Vec<String>) {
    let key_cmd = format!("ASSET_STUDIO_{}_CMD", id.to_uppercase());
    let key_args = format!("ASSET_STUDIO_{}_ARGS", id.to_uppercase());
    if let Ok(cmd) = env::var(&key_cmd) {
        let arg_list: Vec<String> = match env::var(&key_args) {
            Ok(a) => a.split_whitespace().map(String::from).collect(),
            Err(_) => vec![],
        };
        return (cmd, arg_list);
    }
    (program.to_string(), args.to_vec())
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
            program: "python".into(),
            args: vec!["-m".into(), "model_servers.trellis2_paint.server".into()],
            env: vec![("TORCH_CUDA_ARCH_LIST".into(), "8.9".into())],
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
            program: "python".into(),
            args: vec!["-m".into(), "model_servers.hunyuan3d21.server".into()],
            env: vec![("TORCH_CUDA_ARCH_LIST".into(), "8.9".into())],
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
            program: "python".into(),
            args: vec!["-m".into(), "model_servers.hi3d.server".into()],
            env: vec![("TORCH_CUDA_ARCH_LIST".into(), "8.9".into())],
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
            program: "python".into(),
            args: vec!["-m".into(), "model_servers.animation_tools.server".into()],
            env: vec![],
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
    fn spec_defaults_and_url() {
        let s = find_server("trellis2").unwrap();
        assert_eq!(s.url(), "http://127.0.0.1:8123");
        let spec = s.spec();
        assert_eq!(spec.id, "trellis2");
        assert!(spec.uses_gpu);
        assert!(spec.program == "python");
        // Info carries the tasks.
        let info = s.info();
        assert_eq!(info.tasks.len(), s.tasks.len());
    }
}
