# Changelog

All notable changes to `kaos-ml-core` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a1] — 2026-05-08

First public alpha. Pre-release audit pass `audit-01` (8 findings, all
fixed with regression tests where applicable).

### Security

- **KMC-001 (MED) — benchmark tests separated from `tests/unit`.** The
  pytest-benchmark workloads in `tests/unit/test_corpus_benchmarks.py`
  ran by default with the bounded unit command. Moved to
  `tests/benchmarks/test_corpus_benchmarks.py` and re-marked
  `pytest.mark.benchmark` so `pytest tests/unit` no longer pulls in
  benchmark workloads.
- **KMC-002 (MED) — `live` marker registered for credential-bearing tests.**
  `pytest.mark.live` is now declared in `pyproject.toml` and applied
  alongside `integration` to every test file that requires
  `ANTHROPIC_API_KEY` (the four USC vertical-slice tests). Validators
  can now select `unit` / `benchmark` / `integration` / `live` tiers
  independently.

### Changed

- **KMC-003 — Cargo package metadata expanded.** `Cargo.toml` now
  carries `repository`, `homepage`, `documentation`, explicit `readme`,
  `keywords`, `categories`, plus a `[package.metadata.docs.rs]` block
  per `docs/oss/30-rust-packaging/cargo-conventions.md`.
- **KMC-004 — Rust crate-root lints + free-threaded PyO3 module.**
  `rust/lib.rs` declares `#![warn(rust_2018_idioms,
  rust_2021_compatibility, unreachable_pub, unused_qualifications)]`
  per `docs/oss/30-rust-packaging/clippy-and-quality.md`, and the PyO3
  root module is annotated `#[pymodule(gil_used = false)]` for
  free-threaded Python (PEP 703 / cpython-3.14t). v0 exposes only the
  `version()` smoke-test wrapper — no `#[pyclass]` types, no shared
  mutable state. `bindings/{mod,version}.rs` register-fns moved from
  `pub` to `pub(crate)` to satisfy `unreachable_pub`. `missing_docs`
  is allowed at the crate root with a tracking note pending a focused
  docs-backfill.
- **KMC-005 — dev dependency group aligned with the platform standard.**
  Added `pytest-cov>=7.1.0`, bumped `pytest-asyncio>=1.3.0`, bumped
  `pytest>=9.0.3`, pinned `ty>=0.0.34,<0.1` per the cross-package
  baseline established in `kaos-graph 0.1.0a2`.
- **KMC-006 — uv cache keys for the Rust extension.** `[tool.uv]`
  declares `cache-keys = [{ file = "pyproject.toml" }, { file =
  "Cargo.toml" }, { file = "**/*.rs" }]` so uv reliably rebuilds the
  `_rust` extension when Rust sources change.
- **KMC-007 — public API export renamed from `_rust_version` to
  `rust_version`.** The top-level `__all__` no longer exports a
  private-name binding. The underlying value (the Cargo crate version
  string) is unchanged.
- **KMC-008 — `embed_corpus(settings=...)` keyword.** The dense-feature
  helper now accepts a `KaosMLCoreSettings` instance for typed
  injection at the call site; defaults still resolve through
  `KaosMLCoreSettings.resolve(None)` (env / `.env` / field defaults).

### Added

- **Rust core** (`rust/core/`) — pure-Rust skeleton with a `version()`
  smoke test. v2.0+ phases land hot paths (sparse vectorizer assembly,
  sparse scalers, parallel cosine kernels) here.
- **PyO3 bindings** (`rust/bindings/`) — registers `_rust.__version__`
  and `_rust.version()` on the extension module; abi3 wheels target
  Python ≥ 3.13.
- **Python API** (`python/kaos_ml_core/`) — `Corpus`, `CorpusUnit`,
  `CorpusIndex`, `CorpusIndexManifest`, `KaosMLCoreSettings`, plus
  the typed exception hierarchy (`KaosMLCoreError`, `CorpusError`,
  `FeatureError`, `LabelError`, `TrainError`, `PredictError`).
- **v0 vertical slice** — featurize → cluster → LLM-label → train →
  apply, end-to-end on the kaos-content AST. One algorithm per step:
  `BAAI/bge-small-en-v1.5` (via `[transformers]` extra) →
  `MiniBatchKMeans` → `kaos_llm_core.starter.classify` (via `[llm]`
  extra) → `LogisticRegression(solver="liblinear")` → `predict_proba`
  + threshold → `TabularDocument` joined by row index.
- **AST grounding** — five PRD §5 invariants enforced: row indices
  round-trip through `Corpus.row_for` / `Corpus.block_ref_for`;
  `X.shape[0] == len(corpus)`; predictions emit `TabularDocument`
  carrying the AST `block_ref` for every row.
- **CLI** — `kaos-ml` (administrative); `kaos-ml-serve` (placeholder
  for future MCP server).
- Python 3.13 + 3.14 support; `requires-python = ">=3.13"`.

### License

This release is the first to ship under the Apache License 2.0. Earlier
internal versions were proprietary.

[Unreleased]: https://github.com/273v/kaos-ml-core/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/273v/kaos-ml-core/releases/tag/v0.1.0a1
