//! PyO3 binding for `crate::core::version`.

use pyo3::prelude::*;

/// Return the crate version string baked in at compile time.
#[pyfunction]
fn version() -> &'static str {
    crate::core::version::version()
}

/// Register the `version` callable on the parent `_rust` module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
