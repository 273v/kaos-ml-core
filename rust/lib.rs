//! kaos-ml-core: Classical ML primitives for the Kelvin Agentic OS.
//!
//! v0 ships zero Rust functions beyond a `version()` smoke test. The
//! three-layer crate skeleton exists so v2.0+ phases can land hot paths
//! (sparse vectorizer assembly, sparse scalers, parallel cosine kernels)
//! without restructuring the package.
//!
//! See `docs/internal/prd/kaos-ml-core.md` §§4, 7 for the rationale and
//! the dep-budget constraint that governs which crates may be added.

// Audit-01 KMC-004: crate-root lint set per docs/oss/30-rust-packaging/clippy-and-quality.md.
// PyO3 binding registrations consume "unused" core fns, so dead_code stays at allow.
// `missing_docs` is `allow` while v0 docs backfill catches up; restore to `warn` once
// the public Rust API surface (currently a single version() smoke test) is fully documented.
#![allow(dead_code)]
#![allow(missing_docs)]
#![warn(rust_2018_idioms)]
#![warn(rust_2021_compatibility)]
#![warn(unreachable_pub)]
#![warn(unused_qualifications)]

#[cfg(feature = "pyo3")]
mod bindings;
pub mod core;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// The root Python module `kaos_ml_core._rust`.
///
/// Audit-01 KMC-004: declared `gil_used = false` for free-threaded Python
/// (PEP 703 / cpython-3.14t) compatibility. v0 exposes only a `version()`
/// smoke-test wrapper — no `#[pyclass]` types, no shared mutable state.
#[cfg(feature = "pyo3")]
#[pymodule(gil_used = false)]
#[pyo3(name = "_rust")]
fn kaos_ml_core_rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    bindings::version::register_module(m)?;

    // Set __path__ so Python treats this as a package.
    m.setattr("__path__", pyo3::types::PyList::empty(py))?;
    m.setattr("__package__", "kaos_ml_core._rust")?;

    Ok(())
}
