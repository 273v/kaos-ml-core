"""kaos-ml-core: Classical ML primitives for the Kelvin Agentic OS.

The package takes documents from the kaos-content AST through a complete
supervised pipeline — featurize → cluster → LLM-label → train → apply —
and emits predictions that round-trip back to AST ``block_ref``s.

v0 vertical slice picks one algorithm at every step:

    Step 1 (tokenization)    kaos_nlp_core.tokenizer.Tokenizer
    Step 2 (feature matrix)  BAAI/bge-small-en-v1.5 via kaos-nlp-transformers
    Step 3 (clusters)        sklearn.cluster.MiniBatchKMeans
    Step 4 (LLM labels)      kaos_llm_core.starter.classify per k-medoid seed
    Step 5 (train)           sklearn.linear_model.LogisticRegression (liblinear)
    Step 6 (apply)           predict_proba + threshold → TabularDocument

See ``docs/internal/prd/kaos-ml-core.md`` and
``docs/internal/plans/kaos-ml-core-v0.md``.
"""

from kaos_ml_core._rust import __version__ as _rust_version
from kaos_ml_core._version import __version__
from kaos_ml_core.corpus import Corpus, CorpusUnit
from kaos_ml_core.errors import (
    CorpusError,
    FeatureError,
    KaosMLCoreError,
    LabelError,
    PredictError,
    TrainError,
)
from kaos_ml_core.settings import KaosMLCoreSettings

__all__ = [
    "Corpus",
    "CorpusError",
    "CorpusUnit",
    "FeatureError",
    "KaosMLCoreError",
    "KaosMLCoreSettings",
    "LabelError",
    "PredictError",
    "TrainError",
    "__version__",
    "_rust_version",
]
