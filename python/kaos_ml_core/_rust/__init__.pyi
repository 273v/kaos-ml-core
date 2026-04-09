"""Type stubs for the kaos_ml_core._rust extension module.

The Rust crate is intentionally minimal in v0 — it ships a single
``version()`` smoke-test function plus the ``__version__`` attribute.
v2.0+ phases add real hot-path bindings here as they land.
"""

__version__: str

def version() -> str:
    """Return the kaos-ml-core crate version string baked in at compile time."""
    ...
