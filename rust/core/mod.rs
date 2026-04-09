//! Pure-Rust core for kaos-ml-core.
//!
//! No PyO3 dependency — testable via `cargo test --no-default-features`.
//! In v0 this contains a single `version` module to smoke-test the build
//! pipeline. v2.0+ phases land vectorizer assembly, sparse scalers, and
//! parallel cosine kernels here.

pub mod version;
