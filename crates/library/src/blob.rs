//! A content-addressed blob store on local disk.
//!
//! Blobs are addressed by their full SHA-256 digest (64 hex chars). To keep
//! directory fan-out reasonable (ext4 gets slow past ~10k entries per
//! directory), digests are sharded into two-level subdirectories:
//!
//!   `blobs/<digest[0..2]>/<digest[2..4]>/<digest>`
//!
//! Writing is idempotent: re-writing the same digest is a no-op if the file
//! already exists. Reads are a direct path lookup.
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// A content-addressed blob store rooted at a directory.
pub struct BlobStore {
    root: PathBuf,
}

impl BlobStore {
    /// Open (or create) a blob store rooted at `root`.
    pub fn open(root: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// The root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The on-disk path for a digest.
    pub fn path_for(&self, digest: &str) -> PathBuf {
        self.root.join(&digest[0..2]).join(&digest[2..4]).join(digest)
    }

    /// Write `bytes` under `digest` (idempotent). Returns the stored length.
    pub fn write(&self, digest: &str, bytes: &[u8]) -> io::Result<usize> {
        let path = self.path_for(digest);
        if path.exists() {
            return Ok(bytes.len());
        }
        // Create parent dirs, then write atomically: write to a temp file in
        // the same dir, then rename into place (atomic on the same filesystem).
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, bytes)?;
        fs::rename(&tmp, &path)?;
        Ok(bytes.len())
    }

    /// Read the blob for `digest` (expected `size` bytes).
    pub fn read(&self, digest: &str, size: usize) -> io::Result<Vec<u8>> {
        let path = self.path_for(digest);
        let data = fs::read(&path)?;
        if data.len() != size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "blob {digest} has {} bytes, expected {size}",
                    data.len()
                ),
            ));
        }
        Ok(data)
    }

    /// Whether a blob exists for `digest`.
    pub fn contains(&self, digest: &str) -> bool {
        self.path_for(digest).exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store() -> (tempfile::TempDir, BlobStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path().to_path_buf()).unwrap();
        (dir, store)
    }

    #[test]
    fn write_read_roundtrip() {
        let (_d, store) = tmp_store();
        let data = b"hello asset-studio";
        store.write("aaaa", data).unwrap();
        assert_eq!(store.read("aaaa", data.len()).unwrap(), data);
    }

    #[test]
    fn idempotent_write() {
        let (_d, store) = tmp_store();
        let data = b"dedup";
        store.write("bbbb", data).unwrap();
        store.write("bbbb", data).unwrap();
        assert_eq!(store.read("bbbb", data.len()).unwrap(), data);
    }

    #[test]
    fn size_mismatch_errors() {
        let (_d, store) = tmp_store();
        let data = b"12345";
        store.write("cccc", data).unwrap();
        assert!(store.read("cccc", 99).is_err());
    }

    #[test]
    fn missing_blob_errors() {
        let (_d, store) = tmp_store();
        assert!(!store.contains("dddd"));
        assert!(store.read("dddd", 0).is_err());
    }
}
