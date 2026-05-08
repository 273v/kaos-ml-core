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
    """Python ``__version__`` (PEP 440, e.g. ``0.1.0a1``) must map to the
    Rust crate version (Cargo SemVer, e.g. ``0.1.0-alpha.1``) bidirectionally.

    The two are intentionally distinct strings: PyPI distributes PEP 440;
    Cargo publishes SemVer. They are derived from the same source of
    truth (``Cargo.toml [package].version``) by maturin at build time.
    Pre-release tags map ``-alpha.N`` ↔ ``aN``, ``-beta.N`` ↔ ``bN``,
    ``-rc.N`` ↔ ``rcN``.
    """
    import re

    from kaos_ml_core import __version__
    from kaos_ml_core._rust import __version__ as rust_version

    def _normalize(v: str) -> str:
        # Cargo SemVer "X.Y.Z-alpha.N" → PEP 440 "X.Y.ZaN".
        v = re.sub(r"-alpha\.(\d+)", r"a\1", v)
        v = re.sub(r"-beta\.(\d+)", r"b\1", v)
        v = re.sub(r"-rc\.(\d+)", r"rc\1", v)
        return v

    assert _normalize(rust_version) == __version__, (
        f"Python __version__={__version__!r} does not normalize-equal "
        f"Rust __version__={rust_version!r} (normalized: {_normalize(rust_version)!r}). "
        "Fix: bump Cargo.toml [package].version; maturin will produce both forms."
    )
