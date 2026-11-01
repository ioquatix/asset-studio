//! Resolving a job's input bindings to concrete library assets.
//!
//! A [`Job`](crate::Job) references inputs by role as either a ready asset
//! (`InputRef::Asset`) or an output of a dependency job (`InputRes::JobOutput`).
//! By the time a job runs, its dependencies have succeeded and each
//! `JobOutput` resolves to a concrete asset id (from the dependency's
//! `outputs` vector). [`resolve_inputs`] performs that mapping against an
//! asset store and validates that every required role is present.
use crate::{Asset, AssetId, InputRef, Job, TaskInfo};
use std::collections::BTreeMap;

/// A concrete, ready-to-feed input: a role bound to a library asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedInput {
    pub role: String,
    pub asset: Asset,
}

/// Resolve `job`'s inputs to concrete assets, given:
///
/// * `asset_by_id` — look up a live asset by id,
/// * `output_of` — map a dependency `JobId` to its list of output asset-ids.
///
/// Errors with `MissingInput` if a required role is absent, `UnknownAsset` if a
/// referenced asset cannot be found, or `InvalidInput` if a dependency has not
/// produced a usable output.
pub fn resolve_inputs(
    job: &Job,
    task: &TaskInfo,
    asset_by_id: impl Fn(&AssetId) -> Option<Asset>,
    output_of: impl Fn(&crate::JobId) -> Vec<AssetId>,
) -> crate::Result<BTreeMap<String, ResolvedInput>> {
    let mut resolved: BTreeMap<String, ResolvedInput> = BTreeMap::new();

    for (role, input) in &job.inputs {
        let asset = match input {
            InputRef::Asset(ref ref_) => match asset_by_id(&ref_.id) {
                Some(asset) => asset,
                None => return Err(crate::error::Error::UnknownAsset(ref_.id.0.clone())),
            },
            InputRef::JobOutput(ref dep) => {
                let outputs = output_of(&dep.job_id);
                match outputs.get(dep.output_index) {
                    Some(asset_id) => match asset_by_id(asset_id) {
                        Some(asset) => asset,
                        None => return Err(crate::error::Error::UnknownAsset(asset_id.0.clone())),
                    },
                    None => {
                        return Err(crate::error::Error::InvalidInput {
                            task: task.name.clone(),
                            detail: format!(
                                "job {:?}",
                                dep
                            ),
                        })
                    }
                }
            }
        };
        resolved.insert(role.clone(), ResolvedInput { role: role.clone(), asset });
    }

    for (role, _) in &task.input_roles {
        if !resolved.contains_key(role) {
            return Err(crate::error::Error::MissingInput(role.clone()));
        }
    }

    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetKind, JobId};

    fn asset(id: &str) -> Asset {
        Asset {
            id: AssetId::from(id),
            kind: AssetKind::Mesh,
            mime: "model/gltf-binary".into(),
            title: None,
            size_bytes: 4,
            blob_sha256: String::new(),
            rating: None,
            tags: vec![],
            producer: None,
            created_at: 0,
            deleted_at: None,
        }
    }

    #[test]
    fn resolves_asset_inputs() {
        let mut job = Job {
            id: JobId::from("j"),
            server: "s".into(),
            task: "3d-paint".into(),
            state: crate::JobState::Queued,
            prompt: None,
            params: serde_json::json!({}),
            inputs: BTreeMap::new(),
            count: 1,
            seeds: None,
            idempotency_key: None,
            outputs: vec![],
            error: None,
            created_at: 0,
            updated_at: 0,
            retry_of: None,
        };
        let mut inputs = BTreeMap::new();
        inputs.insert("mesh".into(), InputRef::Asset(crate::AssetRef { id: AssetId::from("a1"), role: "mesh".into() }));
        job.inputs = inputs;

        let task = TaskInfo {
            name: "3d-paint".into(),
            description: String::new(),
            input_roles: {
                let mut m = BTreeMap::new();
                m.insert("mesh".into(), vec!["model/gltf-binary".into()]);
                m
            },
            uses_prompt: false,
        };

        let out = resolve_inputs(
            &job,
            &task,
            |id| (id.0 == "a1").then(|| asset("a1")),
            |_| vec![],
        )
        .unwrap();
        assert_eq!(out.get("mesh").unwrap().asset.id, AssetId::from("a1"));
    }
}
