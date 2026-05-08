"""kaos-ml-core: Classical ML primitives for the Kelvin Agentic OS.

The package takes documents from the kaos-content AST through a complete
supervised pipeline — featurize → cluster → LLM-label → split → train →
evaluate → tune-threshold → apply (predict + aggregate) — and emits
predictions that round-trip back to AST ``block_ref``s. Built on top of
``kaos-nlp-transformers`` (dense embeddings), ``kaos-llm-core``
(LLM-driven labeling), and ``kaos-content`` (the AST + TabularDocument
output carrier).

Public surface (0.1.0a1):

  Data:        ``Corpus``, ``CorpusUnit``, ``CorpusIndex``,
               ``CorpusIndexManifest``
  Pipeline:    ``Pipeline`` (with ``save`` / ``load`` / ``predict``)
  Metrics:     ``Metrics``, ``evaluate``, ``wilson_score_interval``
  Splits:      ``SplitResult``, ``stratified_split``
  Threshold:   ``ThresholdResult``, ``tune_threshold``
  Aggregation: ``aggregate_predictions``
  MCP tools:   ``register_ml_tools`` (registers 11 tools on a runtime)
  Settings:    ``KaosMLCoreSettings``
  Errors:      ``KaosMLCoreError``, ``CorpusError``, ``FeatureError``,
               ``LabelError``, ``TrainError``, ``PredictError``

Granularity levels supported by ``Corpus.from_documents(level=...)``:

  - ``"paragraph"`` (default; clause-level classification)
  - ``"sentence"`` (finest)
  - ``"section"`` (group by section_ref; due-diligence)
  - ``"document"`` (one row per doc; ediscovery responsiveness)

For cross-granularity workflows (predict at clause level, decide at
doc level) use :func:`aggregate_predictions`.
"""

from kaos_ml_core._rust import __version__ as rust_version
from kaos_ml_core.aggregate import aggregate_predictions
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
from kaos_ml_core.metrics import Metrics, evaluate, wilson_score_interval
from kaos_ml_core.pipeline import Pipeline, PipelineError
from kaos_ml_core.settings import KaosMLCoreSettings
from kaos_ml_core.split import SplitResult, stratified_split
from kaos_ml_core.threshold import ThresholdResult, tune_threshold

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
    "Metrics",
    "Pipeline",
    "PipelineError",
    "PredictError",
    "SplitResult",
    "ThresholdResult",
    "TrainError",
    "__version__",
    "aggregate_predictions",
    "evaluate",
    "rust_version",
    "stratified_split",
    "tune_threshold",
    "wilson_score_interval",
]
