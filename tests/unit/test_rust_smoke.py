"""Smoke test for the kaos_ml_core._rust extension module.

In v0 the Rust crate ships only a version() function. This test exists
to verify the maturin build pipeline is wired correctly so v2.0+ phases
can land hot paths in the same crate without restructuring.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_rust_module_loads():
    from kaos_ml_core import _rust

    assert _rust.__version__
    assert _rust.__version__.count(".") >= 1


def test_rust_version_function():
    from kaos_ml_core import _rust

    assert _rust.version() == _rust.__version__


def test_rust_version_matches_python_version():
    from kaos_ml_core import __version__
    from kaos_ml_core._rust import __version__ as rust_version

    assert __version__ == rust_version, (
        f"Python version {__version__} != Rust version {rust_version}. "
        "Fix: bump both _version.py and Cargo.toml together."
    )
