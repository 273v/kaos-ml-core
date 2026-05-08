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

from kaos_ml_core._rust import __version__ as rust_version
from kaos_ml_core.corpus import Corpus, CorpusUnit
from kaos_ml_core.errors import (
    CorpusError,
    FeatureError,
    KaosMLCoreError,
    LabelError,
    PredictError,
    TrainError,
)
from kaos_ml_core.index import CorpusIndex, CorpusIndexManifest
from kaos_ml_core.settings import KaosMLCoreSettings

# `__version__` reads from installed package metadata so it matches what
# `pip show kaos-ml-core` reports (PEP 440 form, e.g. "0.1.0a1"). Falling
# back to the Rust extension's Cargo SemVer string ("0.1.0-alpha.1") only
# happens for editable/in-place builds where dist-info is missing — which
# would otherwise cause a version drift between `kaos_ml_core.__version__`
# and PyPI's metadata. See per-package-release.md A7 / F009 lesson #3.
try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__ = _version("kaos-ml-core")
    except _PackageNotFoundError:  # pragma: no cover - source/editable build only
        from kaos_ml_core._rust import __version__
    del _version, _PackageNotFoundError
except Exception:  # pragma: no cover - defensive: importlib.metadata always present on 3.13+
    from kaos_ml_core._rust import __version__

__all__ = [
    "Corpus",
    "CorpusError",
    "CorpusIndex",
    "CorpusIndexManifest",
    "CorpusUnit",
    "FeatureError",
    "KaosMLCoreError",
    "KaosMLCoreSettings",
    "LabelError",
    "PredictError",
    "TrainError",
    "__version__",
    "rust_version",
]
