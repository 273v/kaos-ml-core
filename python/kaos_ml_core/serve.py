"""Stub MCP server entry for kaos-ml-core.

The full MCP server ships in Phase v1.8 with 8-12 tools. v0 ships this
stub so the entry point is registered and ``kaos doctor`` recognizes the
package.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(
        "kaos-ml-serve is not yet implemented. "
        "MCP tools for kaos-ml-core are planned for Phase v1.8. "
        "Use the Python API directly instead."
    )


if __name__ == "__main__":
    main()
