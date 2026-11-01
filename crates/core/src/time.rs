//! Time helpers.

/// The current wall-clock time as seconds since the Unix epoch.
pub fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_is_reasonable() {
        let t = now();
        // After 2020-01-01 and before 2100-01-01.
        assert!(t > 1_577_836_800 && t < 4_102_444_800, "unexpected now: {t}");
    }
}
