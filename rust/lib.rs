//! kaos-ml-core: Classical ML primitives for the Kelvin Agentic OS.
//!
//! v0 ships zero Rust functions beyond a `version()` smoke test. The
//! three-layer crate skeleton exists so v2.0+ phases can land hot paths
//! (sparse vectorizer assembly, sparse scalers, parallel cosine kernels)
//! without restructuring the package.
//!
//! See `docs/internal/prd/kaos-ml-core.md` §§4, 7 for the rationale and
//! the dep-budget constraint that governs which crates may be added.

#![allow(dead_code)]

#[cfg(feature = "pyo3")]
mod bindings;
pub mod core;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// The root Python module `kaos_ml_core._rust`.
#[cfg(feature = "pyo3")]
#[pymodule]
#[pyo3(name = "_rust")]
fn kaos_ml_core_rust(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    bindings::version::register_module(m)?;

    // Set __path__ so Python treats this as a package.
    m.setattr("__path__", pyo3::types::PyList::empty(py))?;
    m.setattr("__package__", "kaos_ml_core._rust")?;

    Ok(())
}
