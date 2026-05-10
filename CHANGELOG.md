# Changelog

All notable changes to `kaos-ml-core` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Per-wheel smoke test in release.yml** — every built wheel now
  imports its target package on its target runner before
  publish-pypi gates. Catches PE/ELF/Mach-O dyld issues, ABI
  mismatches, and missing-runtime-dep surprises that the build step
  alone can't see. Pre-fix, all 5 wheels uploaded blind.

### Removed

- **musllinux wheels (Alpine Linux / musl libc)** dropped from the
  release.yml matrix. ``kaos_ml_core-*-cp313-abi3-musllinux_1_2_x86_64.whl``
  and ``-aarch64.whl`` will not ship on the next release. Family-
  consistency move (matches kaos-nlp-core / kaos-graph /
  kaos-nlp-transformers): downstream sibling kaos-nlp-transformers
  can't ship musllinux because ort's ``download-binaries`` feature
  pulls Microsoft's official libonnxruntime which is glibc-only.
  Shipping musllinux for kaos-ml-core while the ML sibling can't
  install on Alpine creates a fragmented user experience. Pre-1.0
  alpha allows the platform reduction; the 0.1.0a1 release retains
  its musllinux wheels on PyPI for standalone Alpine users
  (``pip install kaos-ml-core==0.1.0a1``).

## [0.1.0a1] — 2026-05-08

First public alpha. Closes the eight `audit-01` findings (KMC-001..008,
documented below) and ships the full classical-ML pipeline for
classification of legal documents at four granularities.

This release was paused mid-prep on 2026-05-08 to address scope gaps
surfaced during a functionality review (the package shipped the
6-step happy path but missed evaluation, threshold tuning, pipeline
persistence, granularity levels beyond paragraph/sentence, and the
agentic MCP surface that the docs claimed). Those gaps are now closed.

### Added

- **`metrics.py`** — `Metrics` (precision, recall, F1, accuracy, ROC
  AUC, confusion matrix, support, **Wilson 95% recall CI**) +
  `evaluate(y_true, y_pred, *, y_proba, classes, confidence)` +
  `wilson_score_interval(positives, total, confidence)`. Wilson 1927
  hand-rolled (no scipy dependency) and tested against tabulated
  reference intervals at the boundary cases. Closes CLAUDE.md hard
  rule #3 ("never report a recall point estimate without a Wilson 95%
  CI").
- **`split.py`** — `SplitResult(train_idx, test_idx, control_idx)` +
  `stratified_split(labels, *, test_frac, control_frac, seed)`.
  Stratified two-step split (peel control, then peel test from the
  remainder) on top of `sklearn.model_selection.StratifiedShuffleSplit`.
  The `control_idx` is what `tune_threshold` consumes — closes
  CLAUDE.md hard rule #5 at the API level.
- **`threshold.py`** — `ThresholdResult` + `tune_threshold(y_true,
  y_proba, *, target_recall, target_precision)`. Sweeps
  `precision_recall_curve` and picks the operating point. Refuses
  hard predictions (0.0/1.0 only) with a `RuntimeWarning` pointing at
  CLAUDE.md hard rule #5. Mutually exclusive targets enforced.
- **`pipeline.py`** — `Pipeline(@frozen, slots)` bundling
  `embed_model_id`, `embed_revision` (pinned), `classifier`,
  `threshold`, `classes`, `kaos_ml_core_version`, `train_metrics`,
  `notes`, `extras`. Methods: `predict(corpus) -> TabularDocument`,
  `predict_proba(corpus) -> np.ndarray`, `save(path)`, `load(path)`.
  Persistence: a directory with `manifest.json` (carrying the magic
  string `kaos-ml-core/pipeline:v1` — load refuses files without it,
  matching the kaos-graph A2-#4 hardening pattern) + `classifier.joblib`.
  Cross-version load: refuses manifests with `format_version >`
  installed kaos-ml-core's; warns on `embed_revision` drift between
  manifest and live registry but does not refuse.
- **`aggregate.py`** — `aggregate_predictions(predictions, *, by,
  method, positive_class)` rolls fine-grained predictions up to a
  coarser key (`doc_uri`, `section_ref`, etc.). Six methods: `any`,
  `all`, `max`, `mean`, `count`, `majority`. Output preserves
  `supporting_block_refs` (semicolon-separated, capped at 50) so a UI
  can drill from doc-level decisions back to the triggering paragraph.
  Built on top of `kaos_content.model.tabular.TabularDocument`.
- **`tools.py`** — `register_ml_tools(runtime)` registers 11 MCP tools
  spanning the full lifecycle: `kaos-ml-build-corpus`,
  `kaos-ml-corpus-info`, `kaos-ml-cluster`,
  `kaos-ml-label-seeds-with-llm`, `kaos-ml-train`, `kaos-ml-evaluate`,
  `kaos-ml-tune-threshold`, `kaos-ml-predict`, `kaos-ml-aggregate`,
  `kaos-ml-save-pipeline`, `kaos-ml-load-pipeline`. Session-scoped
  registries (`_CORPORA`, `_PIPELINES`, `_PREDICTIONS`) keyed by
  `KaosContext.session_id`, same shape as kaos-tabular's `_ENGINES`.
  Tool descriptions explicitly chain prerequisite + follow-up tools so
  an agent can run the full TAR / due-diligence flow end-to-end.
- **`Corpus.from_documents(level="section")`** — group paragraphs by
  `section_ref`; concatenate intra-section paragraphs into one
  CorpusUnit. Joins `level="paragraph"`, `"sentence"`, `"document"`
  for **four supported granularities**, matching the contract-analytics
  / due-diligence / ediscovery use-case grid.
- **`train_logreg`** now accepts EITHER a sparse `dict[int, str]`
  (legacy v0 shape from `label_seeds_with_llm()`) OR a dense
  `np.ndarray | list[str]` aligned with X (the new
  `stratified_split`-driven flow). Backward compatible.
- **23-symbol public API** (was 13). New top-level exports: `Metrics`,
  `Pipeline`, `PipelineError`, `SplitResult`, `ThresholdResult`,
  `aggregate_predictions`, `evaluate`, `stratified_split`,
  `tune_threshold`, `wilson_score_interval`.
- **Optional extras** (all resolvable as of 0.1.0a1): `[transformers]`
  → `kaos-nlp-transformers>=0.1.0a2`, `[llm]` → `kaos-llm-core>=0.1.0a3`,
  `[mcp]` → `kaos-mcp>=0.1.0a2`. No vaporware extras.

### Security (audit-01)

Pre-release audit pass `audit-01` (8 findings, all fixed with
regression tests).

- **KMC-001 (MED) — benchmark tests separated from `tests/unit`.**
  Benchmark workloads moved to `tests/benchmarks/` and re-marked
  `pytest.mark.benchmark` so the bounded unit command no longer pulls
  benchmark workloads.
- **KMC-002 (MED) — `live` marker registered for credential-bearing
  tests.** Live LLM tests get both `integration` and `live` so
  validators can select tiers cleanly.
- **KMC-003 (MED) — Cargo metadata.** `Cargo.toml` carries
  `repository`, `homepage`, `documentation`, `readme`, `keywords`,
  `categories`, plus `[package.metadata.docs.rs]`.
- **KMC-004 (LOW) — Rust crate-root warning lints + free-threaded PyO3
  module.** `rust/lib.rs` declares `#![warn(rust_2018_idioms,
  rust_2021_compatibility, unreachable_pub, unused_qualifications)]`;
  PyO3 root module annotated `#[pymodule(gil_used = false)]`.
- **KMC-005 (LOW) — dev dependency group aligned with the platform
  baseline** (`pytest>=9.0.3`, `pytest-asyncio>=1.3.0`,
  `pytest-cov>=7.1.0`, `ty>=0.0.34`).
- **KMC-006 (LOW) — uv cache keys for the Rust extension.**
  `[tool.uv].cache-keys` covers `pyproject.toml`, `Cargo.toml`, and
  `**/*.rs`.
- **KMC-007 (LOW) — public API export renamed from `_rust_version` to
  `rust_version`.** No private-name binding in `__all__`.
- **KMC-008 (LOW) — `embed_corpus(settings=...)` keyword.** Settings
  injection at the call site.

### Changed

- README rewritten to describe the actual shipped surface — full
  end-to-end Quick Start backed by `tests/integration/test_pipeline_endtoend.py`.
  No vaporware Rust-hot-path paragraph; the Rust crate stays a
  `version()` smoke-test stub in 0.1.0a1 (v2.0+ phases land hot paths
  there as profiling warrants).
- License flipped from `LicenseRef-Proprietary` to `Apache-2.0`
  (PEP 639 string form). LICENSE (`md5 18c184a417afab6dcc2bebdd20e0add1`)
  + NOTICE shipped.
- All five `[project.urls]` declared (Homepage, Documentation,
  Repository, Issues, Changelog) for PyPI page rendering.

### Deferred (honestly documented; v0.1.0a2 / v1.x)

- **TF-IDF / sparse features** — `tfidf_corpus()` exists as a
  `NotImplementedError` stub. v1.1 wires `kaos_nlp_core.search.Searcher`
  + `SparseTermMatrix`.
- **Active learning** — `[al]` extra with `small-text>=2.0`. v1.6.
- **Calibration helpers** — Platt scaling / isotonic regression
  wrappers around sklearn. Users can use `sklearn.calibration.CalibratedClassifierCV`
  directly today.
- **CLI `train` / `evaluate` / `predict` subcommands** — only `info`
  ships in 0.1.0a1. The Python API and MCP tool surface are the
  canonical entry points.
- **Reranker integration in the labeling step** — kaos-nlp-transformers
  ships `CrossEncoderReranker`; we don't yet use it in `label.py`.
  Future: refine the seed-labeling step by reranking the cluster
  representatives.
- **Clause-level granularity** — needs a clause chunker that doesn't
  yet exist. v0 supports `paragraph` / `sentence` / `section` /
  `document` and users with custom chunkers can call
  `Corpus.from_units(...)` to build any granularity manually.

### License

This release is the first to ship under the Apache License 2.0. Earlier
internal versions were proprietary.

[Unreleased]: https://github.com/273v/kaos-ml-core/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/273v/kaos-ml-core/releases/tag/v0.1.0a1
