//! Crate version exposed to Python via the bindings layer.

/// Return the crate version string baked in at compile time.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        let v = version();
        assert!(!v.is_empty());
        assert!(v.contains('.'), "expected semver dotted version, got {v}");
    }
}
