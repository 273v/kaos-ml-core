"""Stub MCP server entry for kaos-ml-core.

The full MCP server ships in Phase v1.8 with 8-12 tools. v0 ships this
stub so the entry point is registered and ``kaos doctor`` recognizes the
package.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print(
        "kaos-ml-core MCP server is not implemented in v0. "
        "It ships in Phase v1.8 — see docs/internal/plans/kaos-ml-core-v0.md.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
