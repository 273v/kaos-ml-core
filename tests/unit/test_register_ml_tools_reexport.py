"""Regression test for audit-04 finding: register_ml_tools top-level re-export.

audit-04/kaos-ml-core.md §23-D flagged that the package docstring
(`__init__.py:20`) and the README concept table (`README.md:199`) both
treat `register_ml_tools` as part of the public surface, but
`__init__.py:72-96` did not re-export it. Callers had to know to
import `from kaos_ml_core.tools import register_ml_tools`, which
contradicts the package's documented contract.

This test pins the corrected surface so a future refactor cannot
quietly drop the re-export.
"""

from __future__ import annotations

import kaos_ml_core


def test_register_ml_tools_is_reexported_at_top_level() -> None:
    """`kaos_ml_core.register_ml_tools` must be the same callable as
    `kaos_ml_core.tools.register_ml_tools`.

    The fix re-exports through the top-level facade. This identity
    assertion catches both the obvious regression (someone removes
    the import) and the silent regression (someone re-defines the name).
    """
    from kaos_ml_core.tools import register_ml_tools

    assert kaos_ml_core.register_ml_tools is register_ml_tools


def test_register_ml_tools_in_dunder_all() -> None:
    """`__all__` is the wildcard-import contract — pin the membership."""
    assert "register_ml_tools" in kaos_ml_core.__all__, (
        "audit-04 regression: 'register_ml_tools' missing from "
        "kaos_ml_core.__all__ even though the package docstring and "
        "README both document it as public"
    )
