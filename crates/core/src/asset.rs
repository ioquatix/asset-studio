use serde::{Deserialize, Serialize};
use std::fmt;

/// A stable identifier for an asset in the library.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AssetId(pub String);

impl fmt::Display for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for AssetId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for AssetId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// The broad category an asset belongs to, used for library grouping and
/// input-role validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    Image,
    Mesh,
    Audio,
    Text,
    Video,
    /// Motion: SMPL-X or SOMA skeleton animation.
    Motion,
}

impl AssetKind {
    /// The MIME types accepted for each kind, in import order.
    pub fn accepted_mime(&self) -> &'static [ &'static str] {
        match self {
            AssetKind::Image => &["image/png", "image/jpeg", "image/webp"],
            AssetKind::Mesh => &["model/gltf-binary", "application/octet-stream"],
            AssetKind::Audio => &["audio/wav", "audio/flac", "audio/ogg", "audio/mpeg"],
            AssetKind::Text => &["text/plain"],
            AssetKind::Video => &["video/mp4", "video/webm"],
            AssetKind::Motion => &["application/octet-stream"],
        }
    }

    #[doc(hidden)]
    pub fn as_str(self) -> &'static str {
        match self {
            AssetKind::Image => "image",
            AssetKind::Mesh => "mesh",
            AssetKind::Audio => "audio",
            AssetKind::Text => "text",
            AssetKind::Video => "video",
            AssetKind::Motion => "motion",
        }
    }
}

impl std::fmt::Display for AssetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for AssetKind {
    type Err = crate::error::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "image" => Ok(AssetKind::Image),
            "mesh" => Ok(AssetKind::Mesh),
            "audio" => Ok(AssetKind::Audio),
            "text" => Ok(AssetKind::Text),
            "video" => Ok(AssetKind::Video),
            "motion" => Ok(AssetKind::Motion),
            other => Err(crate::error::Error::InvalidAssetKind(other.to_owned())),
        }
    }
}

/// A library asset: a file (bytes + metadata) with optional rating and tags.
///
/// An asset is an *output* of a generation job (or an imported file). The
/// on-disk bytes live in the `library` crate; this is the metadata that
/// travels through the API and queue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Asset {
    pub id: AssetId,
    pub kind: AssetKind,
    pub mime: String,
    pub title: Option<String>,
    /// Content length in bytes.
    pub size_bytes: u64,
    /// The SHA-256 hex digest of the stored bytes (content-addressing).
    pub blob_sha256: String,
    /// A 0..=5 rating, if rated.
    pub rating: Option<u8>,
    /// Free-form tags, e.g. `style:cartoon`.
    pub tags: Vec<String>,
    /// The id of the job that produced this asset, if any (None = import).
    pub producer: Option<String>,
    /// Creation timestamp (seconds since epoch).
    pub created_at: i64,
    /// Soft-deletion marker; `None` means live.
    pub deleted_at: Option<i64>,
}

impl Asset {
    /// Whether the asset is visible (not soft-deleted).
    pub fn is_live(&self) -> bool {
        self.deleted_at.is_none()
    }

    /// The stored tags must be well-formed: a non-empty, <=64-char token that
    /// starts with an alphanumeric and contains only alphanumerics plus
    /// `: - _ .` afterwards (no leading dashes, so tags can't be mistaken for
    /// CLI flags).
    pub fn validate_tags(&self) -> Result<(), crate::error::Error> {
        for tag in &self.tags {
            let chars = tag.chars().collect::<Vec<_>>();
            let ok = !chars.is_empty()
                && chars.len() <= 64
                && chars[0].is_ascii_alphanumeric()
                && chars
                    .iter()
                    .skip(1)
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_' | '.'));
            if !ok {
                return Err(crate::error::Error::InvalidTag(tag.clone()));
            }
        }
        Ok(())
    }

    /// Set a rating, enforcing the 0..=5 range.
    pub fn set_rating(&mut self, rating: Option<u8>) -> crate::Result<()> {
        if let Some(r) = rating {
            if r > 5 {
                return Err(crate::error::Error::InvalidRating);
            }
        }
        self.rating = rating;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset() -> Asset {
        Asset {
            id: AssetId::from("a1"),
            kind: AssetKind::Image,
            mime: "image/png".into(),
            title: None,
            size_bytes: 10,
            blob_sha256: String::from("0".repeat(64)),
            rating: None,
            tags: vec![],
            producer: None,
            created_at: 0,
            deleted_at: None,
        }
    }

    #[test]
    fn rating_bounds() {
        let mut a = asset();
        a.set_rating(Some(5)).unwrap();
        assert!(a.set_rating(Some(6)).is_err());
        a.set_rating(None).unwrap();
        assert_eq!(a.rating, None);
    }

    #[test]
    fn tag_validation() {
        let mut a = asset();
        a.tags = vec!["style:cartoon".into(), "-bad".into()];
        assert!(a.validate_tags().is_err());
        a.tags = vec!["style:cartoon".into()];
        assert!(a.validate_tags().is_ok());
    }

    #[test]
    fn kind_mime() {
        assert!(AssetKind::Mesh.accepted_mime().contains(&"model/gltf-binary"));
        assert!(AssetKind::Image.accepted_mime().contains(&"image/png"));
    }
}
