//! PyO3 bindings for kaos-ml-core.
//!
//! Thin wrappers exposing pure-Rust functions in `crate::core` to Python.
//! In v0 this only registers the `version` smoke-test module; v2.0+
//! phases add binding modules per hot path.

pub mod version;
