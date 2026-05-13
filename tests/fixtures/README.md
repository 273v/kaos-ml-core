# kaos-ml-core test fixtures — provenance

Per `docs/oss/50-data-and-fixtures/provenance-policy.md`, every fixture
directory documents the source URL, license, retrieval date, and
SHA-256 of every tracked file.

## Scope of this directory

This directory holds **only Python source files** — there are no
binary data fixtures committed under `kaos-ml-core/tests/fixtures/`.
The actual data corpus that drives the kaos-ml-core acceptance gate
(`usc.jsonl`, 68k+ chapter-level United States Code records) is
**downloaded out-of-tree** into the sibling `kaos-nlp-core` package
by `kaos-nlp-core/tests/fixtures/download_hf_fixtures.py`. See that
package's provenance README for the upstream URL, license, and hash
of the data file itself.

The audit table in the provenance policy lists this directory as
"4 files (3 + `__pycache__`)" — the `__pycache__` entry is interpreter
build cache (`.pyc`) and is not a tracked fixture. The two tracked
files are documented below.

## Per-file manifest

| File | Source | License | Retrieved | SHA-256 |
|---|---|---|---|---|
| `__init__.py` | hand-crafted, 273V (empty package marker) | proprietary, 273 Ventures | 2026-04-09 (first git commit `f7c5a76`) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `usc_corpus.py` | hand-crafted, 273V (USC fixture loader for the kaos-ml-core acceptance gate; reads `kaos-nlp-core/tests/fixtures/usc.jsonl` and wraps records as `ContentDocument`s) | proprietary, 273 Ventures | 2026-04-09 (first git commit `f7c5a76`) | `a83ead7ab3866ccbca6d3b5678368890eb56b411e492cb7c3b9d9622becf948a` |

SHA-256 of `__init__.py` is the canonical empty-file hash because the
file is intentionally zero bytes — it exists only so Python recognises
`tests/fixtures/` as a package for the imports in
`tests/{unit,integration}/...`.

## Provenance of the underlying data

`usc_corpus.py` is **generator code**, not a fixture. At test time it
opens `../../kaos-nlp-core/tests/fixtures/usc.jsonl`, which is sourced
from `data.kl3m.ai` — see `kaos-nlp-core/tests/fixtures/` for the
upstream URL, license, retrieval date, and SHA-256 of `usc.jsonl`.
This split (loader in this package, raw data in the sibling package)
is intentional: the same USC corpus drives multiple downstream test
suites (`kaos-nlp-core`, `kaos-ml-core`, `kaos-nlp-transformers`) and
duplicating the 100s of MB of jsonl into each package would be wasteful.

## Refresh procedure

Both files are hand-crafted source code maintained in-tree. To refresh:

1. Edit the file via a normal PR.
2. Re-run `sha256sum tests/fixtures/<file>` and update the matching
   row above in the same commit.

To refresh the underlying USC data, follow the procedure documented in
`kaos-nlp-core/tests/fixtures/README.md`.

## Confirmations (per provenance policy §"Backfill PR template")

- No file in this directory is customer / privileged / pseudonymized
  content. Both files are hand-crafted 273V source code.
- The dep-license policy's denylist does not apply — both files are
  proprietary 273V code under the same license as the rest of
  `kaos-ml-core`.
