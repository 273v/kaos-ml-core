# kaos-ml-core Development Notes

## Purpose

Classical machine-learning primitives for KAOS. Takes documents from the
`kaos-content` AST through a complete supervised pipeline — featurize →
cluster → LLM-label → train → apply — and emits predictions that round-trip
back to AST `block_ref`s.

Sits between `kaos-nlp-core` (tokenization, inverted index, MinHash) and
`kaos-llm-core` (LLM-driven labeling). Sibling to `kaos-nlp-transformers`
which produces dense embeddings.

## Architecture

Three-layer PyO3/Maturin pattern matching `kaos-nlp-core` and `kaos-graph`:

1. **Rust core** (`rust/core/`) — pure Rust, no PyO3, testable via
   `cargo test --no-default-features`. **In v0 this contains a single
   `version()` function.** v2.0+ phases land hot paths here as profiling
   warrants.
2. **PyO3 bindings** (`rust/bindings/`) — thin wrappers exposing Rust
   functions to Python. v0 registers a `version` smoke-test only.
3. **Python API** (`python/kaos_ml_core/`) — the v0 surface. All actual
   algorithm work lives here. sklearn-compatible.

## Dependencies

**Rust (Cargo.toml) — zero net new vs. kaos-nlp-core / kaos-graph:**
- `pyo3 0.28`, `serde`, `serde_json`, `bincode`, `ahash`, `rayon`
- Anything beyond this set must be justified by measured profiling and
  documented in `docs/internal/prd/kaos-ml-core.md` §7.

**Python (pyproject.toml):**
- `kaos-core`, `kaos-content`, `kaos-nlp-core`
- `numpy`, `scipy`, `scikit-learn>=1.8` (for in-tree HDBSCAN)

**Optional extras:**
- `[transformers]` → `kaos-nlp-transformers` (required for v0 vertical slice)
- `[llm]` → `kaos-llm-core` (required for label_seeds_with_llm)
- `[al]` → `small-text>=2.0` (active learning, v1.6)
- `[bundle]` → `skops>=0.10` (model persistence, v1.7)
- `[mcp]` → `kaos-mcp` (MCP tools, v1.8)

## v0 vertical slice

One algorithm at every step. End-to-end live test on real PDFs.

| Step | v0 algorithm | Module |
|---|---|---|
| 1. Tokenization | `kaos_nlp_core.tokenizer.Tokenizer` (already shipped) | — |
| 2. Feature matrix | `BAAI/bge-small-en-v1.5` via fastembed | `features.embed_corpus` |
| 3. Clusters | `MiniBatchKMeans` on L2-normalized inputs | `cluster.minibatch_kmeans` |
| 4. LLM labels | `kaos_llm_core.starter.classify` per k-medoid seed | `label.label_seeds_with_llm` |
| 5. Train | `LogisticRegression(solver="liblinear", class_weight="balanced")` | `train.train_logreg` |
| 6. Apply | `predict_proba` + threshold → `TabularDocument` | `predict.predict_corpus` |

The single integration test gate is
`tests/integration/test_v0_vertical_slice.py`. It must pass with a live
LLM call against Anthropic Haiku, real PDFs from kaos-pdf fixtures, and
the full AST round-trip verified for randomly sampled rows.

## AST grounding — the central design constraint

Every feature in the package is built on top of the five AST-grounding
invariants in `docs/internal/prd/kaos-ml-core.md` §5:

1. `corpus.unit(r).row == r`
2. `corpus.row_for(corpus.unit(r).block_ref) == r`
3. `corpus.block_ref_for(r) == corpus.unit(r).block_ref`
4. `X.shape[0] == len(corpus)`; row `i` of `X` is the featurization of
   `corpus.unit(i).text`
5. Predictions emit a `TabularDocument` joined by row index, carrying
   the AST `block_ref` for every prediction

The pattern mirrors `kaos_content.search._paragraphs_to_records` and
`kaos_nlp_core.search.Searcher`. Internal int row indices bind to AST
`block_ref`s by position in the units list and round-trip through both
`row_for` and `block_ref_for`.

## Hard rules

1. **Never depend on `cleanlab`** (AGPL-3.0). Reimplement confident
   learning math if needed.
2. **Never use `LogisticRegression(solver="lbfgs")` on sparse high-dim
   text.** Use `liblinear` or `saga`.
3. **Never report a recall point estimate without a Wilson 95% CI.**
   Defensibility requirement (v1.5+).
4. **Never fit a vectorizer on `train + test` combined.**
5. **Never tune the operating threshold on training data.** Use a
   held-out control set.
6. **Never undersample the negative class** before a probabilistic
   classifier.
7. **Never use Random Forest on TF-IDF.** `train_classifier(model="rf")`
   raises `NotImplementedError` with the reasoning.
8. **Never operate on raw serialized text with offset heuristics** when
   AST refs are available — every prediction goes through `block_ref`.
   This rule includes chunking: **never pull in a third-party generic
   text splitter** (`chonkie`, `langchain-text-splitters`,
   `semantic-text-splitter`, etc.) when `kaos_content.chunking.SectionChunker`
   and `kaos_nlp_core.segmentation.segment_sentences` already exist. If
   either is missing a capability you need, **add it to that package**.
   Generic raw-string splitters lose provenance, lose footnote/annotation
   partitioning, and break the AST round-trip — they have no place in
   this platform.
9. **Always emit `TabularDocument` for predictions/results**, not flat
   lists.
10. **Never add a Rust dependency** outside the kaos-nlp-core / kaos-graph
    budget without measured profiling justification.
11. **Never add AGPL/GPL dependencies anywhere.** This is a proprietary
    codebase.
12. **Live integration tests are the quality bar.** Mocked tests are
    supplementary only. Per the platform-wide no-fake-tests rule.
13. **Always use the latest model families in tests.** Check
    `kaos-llm-client/tests/integration/test_live.py` for the current
    model landscape before writing any code that references a model id.

## QA Sequence (mandatory)

```bash
# Rust
cargo fmt
cargo clippy --no-default-features -- -D warnings
cargo test --no-default-features

# Build the extension
maturin develop --release

# Python
ruff format python/ tests/
ruff check --fix python/ tests/
ty check python/ tests/
pytest tests/ -v
```

## Documentation

- PRD: `docs/internal/prd/kaos-ml-core.md`
- v0 plan + per-phase roadmap: `docs/internal/plans/kaos-ml-core-v0.md`
- Sibling package: `docs/internal/prd/kaos-nlp-transformers.md`

When adding a new MCP tool (Phase v1.8+), also update:
`docs/index.md`, `docs/architecture.md`, `docs/reference/mcp-inventory.md`,
and `_KNOWN_TOOL_COUNTS` in `kaos-mcp/kaos_mcp/management/status.py`.
