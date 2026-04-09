"""Stub CLI for kaos-ml-core.

The full CLI ships in Phase v1.8 per ``docs/internal/plans/kaos-ml-core-v0.md``.
v0 ships only ``kaos-ml info`` so the entry point exists and the package
is correctly registered with the platform.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kaos-ml",
        description="Classical ML primitives for the Kelvin Agentic OS",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    info = sub.add_parser("info", help="Show kaos-ml-core settings and status")
    info.add_argument("--json", action="store_true", help="Emit JSON envelope")

    args = parser.parse_args(argv)

    if args.cmd is None or args.cmd == "info":
        from kaos_ml_core import __version__
        from kaos_ml_core._rust import __version__ as rust_version
        from kaos_ml_core.settings import KaosMLCoreSettings

        s = KaosMLCoreSettings()
        payload = {
            "command": "info",
            "package": "kaos-ml-core",
            "version": __version__,
            "rust_version": rust_version,
            "default_embed_model": s.default_embed_model,
            "default_threshold": s.default_threshold,
            "recall_target": s.recall_target,
            "recall_confidence": s.recall_confidence,
            "profile": s.profile,
        }
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2))
        else:
            for k, v in payload.items():
                print(f"{k}: {v}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
