# kaos-ml-core

> **Part of [Kelvin Agentic OS](https://kelvin.legal) (KAOS)** — open agentic
> infrastructure for legal work, built by
> [273 Ventures](https://273ventures.com).
> See the [full KAOS package map](https://github.com/273v) for the rest of the stack.

[![PyPI - Version](https://img.shields.io/pypi/v/kaos-ml-core)](https://pypi.org/project/kaos-ml-core/)
[![Python](https://img.shields.io/pypi/pyversions/kaos-ml-core)](https://pypi.org/project/kaos-ml-core/)
[![License](https://img.shields.io/pypi/l/kaos-ml-core)](https://github.com/273v/kaos-ml-core/blob/main/LICENSE)
[![CI](https://github.com/273v/kaos-ml-core/actions/workflows/ci.yml/badge.svg)](https://github.com/273v/kaos-ml-core/actions/workflows/ci.yml)

`kaos-ml-core` is the classical-ML layer of KAOS — a typed Python API
that takes documents from the [`kaos-content`](https://github.com/273v/kaos-content)
AST through a complete supervised pipeline (featurize → cluster →
LLM-label → split → train → evaluate → tune-threshold → predict →
aggregate) and emits predictions that round-trip back to the source
document via stable `block_ref`s.

It is built **on top of**:
[`kaos-content`](https://github.com/273v/kaos-content) (AST + TabularDocument),
[`kaos-nlp-transformers`](https://github.com/273v/kaos-nlp-transformers) (dense
embeddings, L2-normalized output as of 0.1.0a2),
[`kaos-llm-core`](https://github.com/273v/kaos-llm-core) (LLM-driven
labeling), and [`kaos-mcp`](https://github.com/273v/kaos-mcp)
(11-tool agentic surface).

It is dependency-light at the BASE: the install pulls only the core
KAOS runtime (`kaos-core`, `kaos-content`, `kaos-nlp-core`) plus
`numpy`, `scipy`, `scikit-learn`, and `joblib`. Optional extras layer
in the rest of the pipeline — `[transformers]` for fastembed-backed
featurization, `[llm]` for LLM-driven cold-start labeling, and `[mcp]`
for the MCP tool surface.

The Rust crate is intentionally a stub in 0.1.0a1 — only a `version()`
smoke test, no hot path. v2.0+ phases land sparse-vectorizer and
parallel-cosine kernels there as profiling warrants. We don't claim
otherwise.

## Install

```bash
uv add "kaos-ml-core[transformers,llm,mcp]"   # full pipeline
# or
pip install "kaos-ml-core[transformers,llm,mcp]"
```

`kaos-ml-core` requires Python **3.13** or newer. The published wheels
are `cp313-abi3` — one wheel per OS/arch covers every CPython 3.13+
minor (3.13, 3.14, 3.15, …).

Without the optional extras, you can still use `Corpus`, `evaluate`,
`stratified_split`, `tune_threshold`, `aggregate_predictions`, and the
classifier directly with your own feature matrix — the BASE install is
useful on its own for downstream code that already produces embeddings.

## Quick start

This is the full real pipeline: featurize → split → train → evaluate →
tune threshold → save pipeline → load + predict → aggregate to
document level. Backed by the live integration test
`tests/integration/test_pipeline_endtoend.py` against the same fixture.

```python
import numpy as np
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata, SourceRef

from kaos_ml_core import (
    Corpus, Pipeline,
    aggregate_predictions, evaluate, stratified_split, tune_threshold,
)
from kaos_ml_core.features import embed_corpus
from kaos_ml_core.train import train_logreg

# A tiny fixture: 3 contracts, mix of arbitration + non-arbitration
# clauses. In a real flow this would come from kaos-pdf or kaos-office.
arbitration = [
    "Any dispute shall be resolved by binding arbitration in Delaware.",
    "The parties agree to submit any controversy to AAA arbitration.",
    "All claims shall be settled by binding arbitration under JAMS rules.",
    "The parties consent to binding arbitration of any contract dispute.",
]
other = [
    "The Seller indemnifies the Buyer against third-party claims.",
    "Force majeure clauses excuse performance during natural disasters.",
    "This agreement is governed by the laws of New York.",
    "The buyer agrees to make payments within 30 days of invoice.",
    "Indemnification is subject to a cap of 10 percent of purchase price.",
    "Notice of breach must be provided in writing.",
    "Confidential information must not be disclosed.",
    "The agreement may be amended only by written consent.",
]


def _doc(uri: str, paragraphs: list[str]) -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(source=SourceRef(uri=uri)),
        body=tuple(Paragraph(children=(Text(value=p),)) for p in paragraphs),
    )


docs = [
    _doc("contract://A", arbitration[:2] + other[:3]),
    _doc("contract://B", arbitration[2:] + other[3:6]),
    _doc("contract://C", other[6:]),
]
corpus = Corpus.from_documents(docs, level="paragraph")
labels = np.array([
    "arbitration" if corpus.unit(i).text in arbitration else "other"
    for i in range(len(corpus))
])

# 1. Featurize. Embeddings are L2-normalized (kaos-nlp-transformers KNT-101).
X = embed_corpus(corpus)

# 2. Stratified train/test/control split. The control_frac is held out
#    for threshold tuning (CLAUDE.md hard rule #5).
split = stratified_split(labels, test_frac=0.25, control_frac=0.20, seed=42)

# 3. Train logistic regression on the train slice.
clf = train_logreg(X[split.train_idx], labels[split.train_idx])

# 4. Evaluate on the test slice. evaluate() returns precision, recall, F1,
#    accuracy, ROC AUC, confusion matrix, AND a Wilson 95% CI on recall
#    (CLAUDE.md hard rule #3).
test_proba = clf.predict_proba(X[split.test_idx])[:, list(clf.classes_).index("arbitration")]
metrics = evaluate(
    labels[split.test_idx],
    clf.predict(X[split.test_idx]),
    y_proba=test_proba,
    classes=("other", "arbitration"),
)
print(f"F1={metrics.f1:.3f}  recall={metrics.recall:.3f}  ROC AUC={metrics.roc_auc:.3f}")
print(f"Wilson 95% recall CI: [{metrics.recall_ci_lower:.3f}, {metrics.recall_ci_upper:.3f}]")

# 5. Tune the operating threshold on the held-out control set.
control_proba = clf.predict_proba(X[split.control_idx])[
    :, list(clf.classes_).index("arbitration")
]
tuned = tune_threshold(
    labels[split.control_idx], control_proba,
    target_recall=0.85, pos_label="arbitration",
)
print(f"threshold={tuned.threshold:.3f} (control recall={tuned.achieved_recall:.3f})")

# 6. Bundle into a Pipeline and save it. The .kaos directory is portable
#    across hosts — ship to prod with one path.
pipeline = Pipeline(
    embed_model_id="BAAI/bge-small-en-v1.5",
    embed_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    classifier=clf,
    threshold=tuned.threshold,
    classes=("other", "arbitration"),
    kaos_ml_core_version="0.1.0a1",
    train_metrics=metrics,
)
pipeline.save("/tmp/contracts-arbitration-v1.kaos")

# 7. Load the pipeline back and apply to the full corpus.
loaded = Pipeline.load("/tmp/contracts-arbitration-v1.kaos")
predictions = loaded.predict(corpus)
# predictions is a TabularDocument: one row per CorpusUnit with
# block_ref, doc_uri, predicted_label, score, above_threshold.

# 8. Aggregate paragraph-level predictions up to document level. This
#    is the central operation for due-diligence / contract analytics.
agg = aggregate_predictions(
    predictions, by="doc_uri", method="any", positive_class="arbitration",
)
for row in agg.tables[0].rows:
    print(f"{row[0]}: arbitration={row[6]} ({row[2]}/{row[1]} positive paragraphs)")
# contract://A: arbitration=True   (2/5 positive paragraphs)
# contract://B: arbitration=True   (2/6 positive paragraphs)
# contract://C: arbitration=False  (0/2 positive paragraphs)
```

For an agent-driven flow over the same operations, see the [MCP tool
surface](#mcp-tool-surface) below — `register_ml_tools(runtime)` exposes
all 11 lifecycle steps as MCP tools.

## Concepts

The package is built around a small set of typed primitives. Build on
top of [`kaos-content`](https://github.com/273v/kaos-content) for the
AST and [`kaos-nlp-transformers`](https://github.com/273v/kaos-nlp-transformers)
for embeddings.

| Concept | What it is |
|---|---|
| **`Corpus`** | Frozen, AST-grounded set of text units with bidirectional mapping between internal row indices and `block_ref`s. Built from one or more `ContentDocument` instances at four granularities — `"paragraph"` (default), `"sentence"`, `"section"` (group by `section_ref`), `"document"` (one row per doc). Cross-granularity workflows use `aggregate_predictions` to roll up. |
| **`Pipeline`** | Bundles `(embed_model_id, embed_revision, classifier, threshold, classes, kaos_ml_core_version)` with `save(path)` / `load(path)`. Persistence uses `joblib` for the classifier + JSON manifest with a magic-byte header (`load` refuses files without it). `Pipeline.predict(corpus)` does featurization + classification + threshold in one call. |
| **`Metrics` + `evaluate`** | Precision / recall / F1 / accuracy / ROC AUC / confusion matrix on a held-out set, **plus a Wilson 95% CI on recall** (CLAUDE.md hard rule #3). `wilson_score_interval(positives, total, confidence)` is exposed as a public helper for callers that already have raw counts. |
| **`SplitResult` + `stratified_split`** | Three-way stratified split (train / test / control). `control_idx` is the held-out subset that `tune_threshold` consumes — closes CLAUDE.md hard rule #5 ("never tune on the training set") at the API level. |
| **`ThresholdResult` + `tune_threshold`** | Find the operating threshold that hits a target recall (or precision) on a control set. Refuses degenerate inputs (hard predictions) with a fix-the-data warning. |
| **`aggregate_predictions`** | Cross-granularity bridge. Takes the `TabularDocument` from `Pipeline.predict()` and rolls up by `doc_uri` / `section_ref` / any other key with `method ∈ {any, all, max, mean, count, majority}`. Output preserves supporting `block_ref`s so a UI can drill from doc-level decision back to the triggering paragraph. |
| **`register_ml_tools(runtime)`** | Registers the 11-tool MCP surface. Session-scoped registries for corpora / pipelines / predictions; same shape as kaos-tabular's `_ENGINES`. Tool descriptions chain explicitly ("call X first; pass output to Y") so an agent can run the pipeline end-to-end without external orchestration. |
| **`KaosMLCoreSettings`** | Typed settings (env prefix `KAOS_ML_CORE_`) — `default_embed_model`, `default_threshold`, `recall_target`, `recall_confidence`, `profile`. Resolves through the standard six-level KAOS settings hierarchy. |

## CLI

`kaos-ml-core` ships a stub `kaos-ml info` administrative CLI in 0.1.0a1.
The `train` / `evaluate` / `predict` subcommands land in 0.1.0a2 (the
Python API and MCP tool surface are the canonical entry points today).

```bash
kaos-ml info --json    # version + settings + Rust extension status
```

## MCP tool surface

Importing `register_ml_tools(runtime)` registers 11 MCP tools spanning
the full classifier lifecycle:

```python
from kaos_core import KaosRuntime
from kaos_ml_core.tools import register_ml_tools

rt = KaosRuntime()
n = register_ml_tools(rt)
assert n == 11
```

The tools (in lifecycle order):

| Tool | What it does |
|---|---|
| `kaos-ml-build-corpus` | Build a Corpus from ContentDocuments at a chosen granularity. |
| `kaos-ml-corpus-info` | Read-only stats. |
| `kaos-ml-cluster` | MiniBatchKMeans + k-medoid seed selection (caches the feature matrix). |
| `kaos-ml-label-seeds-with-llm` | LLM-driven cold-start labeling via kaos-llm-core. |
| `kaos-ml-train` | Stratified split + train + evaluate; returns `pipeline_id` + Wilson-CI metrics. |
| `kaos-ml-evaluate` | Read-only evaluation on a held-out subset. |
| `kaos-ml-tune-threshold` | Tune operating threshold on the control set. |
| `kaos-ml-predict` | Apply pipeline → predictions table (TabularDocument). |
| `kaos-ml-aggregate` | Roll up predictions to `doc_uri` / `section_ref`. |
| `kaos-ml-save-pipeline` | Persist pipeline to disk. |
| `kaos-ml-load-pipeline` | Load + register a saved pipeline. |

Tool descriptions reference prerequisite + follow-up tools so an agent
can chain them end-to-end.

## Use cases

`kaos-ml-core` was designed for the legal-tech workflows where
predict-at-fine-granularity → aggregate-to-coarse-decision is the
central operation. Pick the granularity that matches the partner's
question:

| Use case | Predict at | Aggregate to | Example aggregation |
|---|---|---|---|
| **Contract analytics** ("find arbitration clauses") | paragraph | `doc_uri` | `method="any"` — list contracts that contain the language |
| **Contract analytics** ("classify doc type: NDA / SPA / lease") | document | — | direct doc-level classification |
| **Due diligence** ("find indemnification sections") | section | `doc_uri` | `method="any"` + supporting `block_ref`s for review |
| **Due diligence** ("triage 5,000 docs into financials/IP/leases") | document | — | direct doc-level classification |
| **TAR / ediscovery** ("responsive vs non-responsive") | document or paragraph | `doc_uri` (when paragraph-level) | `method="any"` for production decisions; `method="count"` for review queues |
| **Privilege detection** | section / paragraph | `doc_uri` | `method="any"` + supporting refs feeds the privilege log |

## Compatibility & status

| Aspect | |
|---|---|
| **Python** | 3.13, 3.14 (informational matrix entries for 3.14t free-threaded and 3.15-dev). One `cp313-abi3` wheel per OS/arch covers all 3.13+ minors. |
| **OS** | Linux (manylinux + musllinux, x86_64 + aarch64), macOS arm64, Windows x86_64, Windows arm64. macOS x86_64 deliberately skipped (Apple ended Intel sales in 2023). |
| **Maturity** | Alpha. The public API is documented in `kaos_ml_core.__all__` (23 symbols). |
| **Stability policy** | Pre-1.0: minor bumps may change behaviour. Every change is documented in [`CHANGELOG.md`](CHANGELOG.md). |
| **Test coverage** | 99 unit tests + integration tests (live LLM round-trip + real-PDF fixtures via kaos-pdf). |
| **Type checker** | Validated with [`ty`](https://docs.astral.sh/ty/), Astral's Python type checker. |

## Companion packages

`kaos-ml-core` is one of the packages in the
[Kelvin Agentic OS](https://kelvin.legal). The broader stack:

| Package | Layer | What it does |
|---|---|---|
| [`kaos-core`](https://github.com/273v/kaos-core) | Core | Foundational runtime, MCP-native types, registries, execution engine, VFS |
| [`kaos-content`](https://github.com/273v/kaos-content) | Core | Typed document AST: Block/Inline, provenance, views |
| [`kaos-mcp`](https://github.com/273v/kaos-mcp) | Bridge | FastMCP server, `kaos` management CLI, MCP resource templates |
| [`kaos-pdf`](https://github.com/273v/kaos-pdf) | Extraction | PDF → AST with provenance |
| [`kaos-web`](https://github.com/273v/kaos-web) | Extraction | Web extraction, browser automation, search, domain intelligence |
| [`kaos-office`](https://github.com/273v/kaos-office) | Extraction | DOCX / PPTX / XLSX readers + writers to AST |
| [`kaos-tabular`](https://github.com/273v/kaos-tabular) | Extraction | DuckDB-powered SQL analytics |
| [`kaos-source`](https://github.com/273v/kaos-source) | Data | Government + financial data connectors (Federal Register, eCFR, EDGAR, GovInfo, PACER, GLEIF) |
| [`kaos-llm-client`](https://github.com/273v/kaos-llm-client) | LLM | Multi-provider LLM transport |
| [`kaos-llm-core`](https://github.com/273v/kaos-llm-core) | LLM | Typed LLM programming (Signatures, Programs, Optimizers) |
| [`kaos-nlp-core`](https://github.com/273v/kaos-nlp-core) | Primitives (Rust) | High-performance NLP primitives |
| [`kaos-nlp-transformers`](https://github.com/273v/kaos-nlp-transformers) | ML | Dense embeddings + retrieval |
| [`kaos-graph`](https://github.com/273v/kaos-graph) | Primitives (Rust) | Graph algorithms + RDF/SPARQL |
| [`kaos-ml-core`](https://github.com/273v/kaos-ml-core) | Primitives (Rust) | Classical ML on the document AST |
| [`kaos-citations`](https://github.com/273v/kaos-citations) | Legal | Legal citation extraction, resolution, verification |
| [`kaos-agents`](https://github.com/273v/kaos-agents) | Agentic | Agent runtime, memory, recipes |
| [`kaos-reference`](https://github.com/273v/kaos-reference) | Sample | Reference module for module authors |

Packages depend on `kaos-core`; everything else is opt-in. Mix and match the
ones you need.

## Development

```bash
git clone https://github.com/273v/kaos-ml-core
cd kaos-ml-core
uv sync --group dev --extra transformers --extra llm --extra mcp
uv run maturin develop --release
```

Install pre-commit hooks (recommended — they run the same checks as CI on
every commit, scoped to staged files):

```bash
uvx pre-commit install
uvx pre-commit run --all-files     # one-time full sweep
```

Manual QA commands (the same set CI runs):

```bash
cargo fmt --check
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --no-default-features --lib
uv run ruff format --check python/kaos_ml_core tests
uv run ruff check python/kaos_ml_core tests
uv run ty check python/kaos_ml_core tests
uv run pytest tests/ -m "not live and not benchmark"
```

## Build from source

```bash
uv build
uv pip install dist/*.whl
```

## Contributing

Issues and pull requests are welcome. By contributing you certify the
[Developer Certificate of Origin v1.1](https://developercertificate.org/) —
sign every commit with `git commit -s`. Please open an issue before starting
on a non-trivial change so we can align on scope.

## Security

For security issues, **please do not file a public issue**. Report privately
via [GitHub Private Vulnerability Reporting](https://github.com/273v/kaos-ml-core/security/advisories/new)
or email **security@273ventures.com**. See [SECURITY.md](SECURITY.md) for the
full disclosure policy.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2026 [273 Ventures LLC](https://273ventures.com).
Built for [kelvin.legal](https://kelvin.legal).
