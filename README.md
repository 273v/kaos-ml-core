# kaos-ml-core

> **Part of [Kelvin Agentic OS](https://kelvin.legal) (KAOS)** — open agentic
> infrastructure for legal work, built by
> [273 Ventures](https://273ventures.com).
> See the [full KAOS package map](https://github.com/273v) for the rest of the stack.

[![PyPI - Version](https://img.shields.io/pypi/v/kaos-ml-core)](https://pypi.org/project/kaos-ml-core/)
[![Python](https://img.shields.io/pypi/pyversions/kaos-ml-core)](https://pypi.org/project/kaos-ml-core/)
[![License](https://img.shields.io/pypi/l/kaos-ml-core)](https://github.com/273v/kaos-ml-core/blob/main/LICENSE)
[![CI](https://github.com/273v/kaos-ml-core/actions/workflows/ci.yml/badge.svg)](https://github.com/273v/kaos-ml-core/actions/workflows/ci.yml)

`kaos-ml-core` is the classical-ML layer for KAOS — a typed Python API
that takes documents from the [`kaos-content`](https://github.com/273v/kaos-content)
AST through a complete supervised pipeline (featurize → cluster →
LLM-label → train → apply) and emits predictions that round-trip back to
the source document via stable `block_ref`s. Algorithms are scikit-learn
where it makes sense (`MiniBatchKMeans`, `LogisticRegression`); the
heavy lifting (sparse vectorizer assembly, parallel cosine kernels) is
reserved for a Rust+PyO3 hot path that v0 leaves stubbed.

It is dependency-light at the BASE: the install pulls only the core
KAOS runtime (`kaos-core`, `kaos-content`, `kaos-nlp-core`) plus
`numpy`, `scipy`, and `scikit-learn`. Optional extras layer in the
rest of the KAOS ecosystem — LLM-driven labeling (`[llm]`) and MCP
tool registration (`[mcp]`) — without adding weight to the BASE.

## Install

```bash
uv add kaos-ml-core
# or
pip install kaos-ml-core
```

`kaos-ml-core` requires Python **3.13** or newer. The published wheels
are `cp313-abi3` — one wheel per OS/architecture covers every CPython
3.13+ minor (3.13, 3.14, 3.15, …). No re-release needed when 3.15 ships.

Platform coverage: Linux x86_64 (manylinux + musllinux), Linux aarch64
(manylinux + musllinux), macOS arm64, Windows x86_64, Windows arm64.

## Quick start

```python
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata, SourceRef
from kaos_ml_core import Corpus

# Build a tiny ContentDocument with three paragraphs.
doc = ContentDocument(
    metadata=DocumentMetadata(source=SourceRef(uri="example://memo/1")),
    body=(
        Paragraph(children=(Text(value="Force majeure clauses excuse performance."),)),
        Paragraph(children=(Text(value="Indemnity caps the liability of the seller."),)),
        Paragraph(children=(Text(value="The buyer agrees to mediation in Delaware."),)),
    ),
)

# Build an AST-grounded Corpus.
corpus = Corpus.from_paragraphs(doc)

# AST-grounding invariants (PRD §5):
assert len(corpus) == 3
assert corpus.unit(0).row == 0
assert corpus.row_for(corpus.unit(2).block_ref) == 2
assert corpus.block_ref_for(1) == corpus.unit(1).block_ref
print(corpus.unit(0).text, "→", corpus.unit(0).block_ref)
# "Force majeure clauses excuse performance. → #/body/0"
```

The full v0 pipeline (featurize → cluster → LLM-label → train → apply)
becomes available with the optional extras in `0.1.0a2` once the
sibling `kaos-nlp-transformers` package publishes. See the
[v0 vertical slice test](https://github.com/273v/kaos-ml-core/blob/main/tests/integration/test_v0_vertical_slice.py)
for the end-to-end shape.

## Concepts

The package is built around a small set of typed primitives.

| Concept | What it is |
|---|---|
| **`Corpus`** | Frozen, AST-grounded set of text units with bidirectional mapping between internal row indices and `block_ref`s. Built from one or more `ContentDocument` instances at paragraph or sentence granularity. Every feature matrix `X` produced from a Corpus satisfies `X.shape[0] == len(corpus)`; row `i` is the featurization of `corpus.unit(i).text`. |
| **`CorpusUnit`** | Frozen dataclass — `row`, `text`, `block_ref`, `doc_uri`, `page`, `section_ref`, `section_title`. The atomic unit of any kaos-ml-core operation. |
| **`CorpusIndex`** | VFS-backed retrievable index over a Corpus. Wraps `kaos-nlp-core`'s `Searcher` and exposes save/load with a versioned `CorpusIndexManifest` (manifest carries the package version, hash, and embed-model id). |
| **Pipeline (v0)** | One algorithm per step: tokenization via `kaos_nlp_core.tokenizer.Tokenizer`; dense embeddings via `BAAI/bge-small-en-v1.5` (`[transformers]` extra, deferred to `0.1.0a2`); clusters via `sklearn.cluster.MiniBatchKMeans`; cold-start labels via `kaos_llm_core.starter.classify` (`[llm]` extra) per k-medoid seed; classifier via `sklearn.linear_model.LogisticRegression(solver="liblinear")`; output is a `TabularDocument` joined by row index. |
| **AST grounding** | Five invariants (PRD §5) enforced at construction time and re-asserted by tests: `unit(r).row == r`; `row_for(unit(r).block_ref) == r`; `block_ref_for(r) == unit(r).block_ref`; `X.shape[0] == len(corpus)`; predictions emit a `TabularDocument` carrying the `block_ref` for every row. |
| **Settings** | `KaosMLCoreSettings` (env prefix `KAOS_ML_CORE_`) — `default_embed_model`, `default_threshold`, `recall_target`, `recall_confidence`, `profile`. Resolves through the standard six-level KAOS settings hierarchy. |
| **Rust core** | A pure-Rust skeleton (`rust/core/`) with a `version()` smoke-test wrapper. Land hot paths (sparse vectorizer assembly, sparse scalers, parallel cosine kernels) here as profiling warrants — the v2.0+ phases reserve the layout. |

## CLI

`kaos-ml-core` ships a `kaos-ml` administrative CLI plus an optional
`kaos-ml-serve` placeholder for a future MCP server (the `[mcp]` extra
will wire it up once the corresponding tool surface lands):

```bash
kaos-ml --json version       # version info (Python + Rust)
kaos-ml-serve                # placeholder; full MCP wiring lands in 0.1.0a2+
```

## Compatibility & status

| Aspect | |
|---|---|
| **Python** | 3.13, 3.14 (informational matrix entries for 3.14t free-threaded and 3.15-dev). One `cp313-abi3` wheel per OS/arch covers all 3.13+ minors. |
| **OS** | Linux (manylinux + musllinux, x86_64 + aarch64), macOS arm64, Windows x86_64, Windows arm64. macOS x86_64 deliberately skipped (Apple ended Intel sales in 2023). |
| **Maturity** | Alpha. The public API is documented in `kaos_ml_core.__all__`. |
| **Stability policy** | Pre-1.0: minor bumps may change behaviour. Every change is documented in [`CHANGELOG.md`](CHANGELOG.md). |
| **Test coverage** | 99 Python unit tests + 9 integration tests (4 of which require a live `ANTHROPIC_API_KEY` for the v0 vertical slice) + 1 Rust unit test. The `[transformers]` extra is deferred to `0.1.0a2`; tests using it skip cleanly when the sibling package is not installed. |
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
uv sync --group dev
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
