# kaos-ml-core

Classical machine learning primitives for the Kelvin Agentic OS — featurize,
cluster, LLM-label, train, and apply classifiers, all grounded back to the
`kaos-content` document AST.

**Status:** v0 scaffold (proposed PRD + plan landed).

## What it does

Take a folder of PDFs (or any other source `kaos-pdf`, `kaos-web`,
`kaos-office` can extract) and produce a `TabularDocument` with one
prediction row per AST paragraph (or sentence). Every prediction
round-trips back to a `block_ref` in the source document.

```python
from kaos_pdf import extract_pdf
from kaos_ml_core import Corpus
from kaos_ml_core.features import embed_corpus
from kaos_ml_core.cluster import minibatch_kmeans, kmedoid_seeds
from kaos_ml_core.label import label_seeds_with_llm
from kaos_ml_core.train import train_logreg
from kaos_ml_core.predict import predict_corpus

# 1-2. Build a Corpus and a feature matrix
docs = [extract_pdf(p) for p in pdf_paths]
corpus = Corpus.from_documents(docs, level="paragraph")
X = embed_corpus(corpus, model="BAAI/bge-small-en-v1.5")

# 3. Initial clusters
clusters = minibatch_kmeans(X, n_clusters=20)
seeds = kmedoid_seeds(X, clusters, per_cluster=3)

# 4. LLM labels for the cold-start seed set
labels = await label_seeds_with_llm(
    corpus, seeds,
    classes=["responsive", "non_responsive"],
    instructions="Is this paragraph responsive to a discovery request about X?",
    model="claude-haiku-4-5",
)

# 5-6. Train and apply
clf = train_logreg(X, labels)
predictions = predict_corpus(corpus, X, clf, threshold=0.5)
# predictions: TabularDocument joined back to block_refs
```

## Use cases

- **Ediscovery / TAR** (Technology-Assisted Review)
- **Contract diligence** — clause classification, change-of-control flagging
- **Regulatory triage** — Federal Register / eCFR review
- **Topic routing** — automatic file labeling for case management

## Install

```bash
# Core install (no dense embeddings, no LLM labeling)
uv add kaos-ml-core

# v0 vertical slice — needs both extras
uv add "kaos-ml-core[transformers,llm]"
```

## Layout

```
kaos-ml-core/
├── Cargo.toml              # Rust crate (zero net new deps vs kaos-nlp-core)
├── pyproject.toml          # maturin build backend
├── rust/
│   ├── lib.rs              # PyO3 module entry
│   ├── core/               # pure Rust (v0: version smoke-test only)
│   └── bindings/           # PyO3 wrappers (v0: version only)
├── python/kaos_ml_core/
│   ├── __init__.py         # public API
│   ├── corpus.py           # Corpus + CorpusUnit (the heart)
│   ├── features.py         # embed_corpus
│   ├── cluster.py          # minibatch_kmeans, kmedoid_seeds
│   ├── label.py            # label_seeds_with_llm
│   ├── train.py            # train_logreg, train_classifier
│   ├── predict.py          # predict_corpus
│   ├── settings.py         # KaosMLCoreSettings
│   └── errors.py
└── tests/
    ├── unit/
    └── integration/        # live PDF + LLM end-to-end test
```

## Documentation

- **PRD** — `docs/internal/prd/kaos-ml-core.md`
- **v0 plan + per-phase roadmap** — `docs/internal/plans/kaos-ml-core-v0.md`
- **Sibling package** — `kaos-nlp-transformers` (dense embeddings)

## Build

```bash
# Rust
cargo fmt && cargo clippy --no-default-features -- -D warnings
cargo test --no-default-features

# Build extension
maturin develop --release

# Python QA
ruff format python/ tests/
ruff check --fix python/ tests/
ty check python/ tests/
pytest tests/ -v
```

## License

LicenseRef-Proprietary © 273 Ventures LLC
