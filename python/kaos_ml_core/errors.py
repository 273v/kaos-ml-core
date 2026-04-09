"""Error hierarchy for kaos-ml-core.

All exceptions inherit from ``KaosCoreError`` (kaos-core) so they
participate in the agent-friendly triplet contract: every error message
must answer (1) what went wrong, (2) how to fix it, (3) alternative
approach when applicable.
"""

from __future__ import annotations

from kaos_core.exceptions import KaosCoreError


class KaosMLCoreError(KaosCoreError):
    """Base error for kaos-ml-core."""


class CorpusError(KaosMLCoreError):
    """Invalid Corpus construction (empty input, missing doc URI, etc.)."""


class FeatureError(KaosMLCoreError):
    """Vectorization failure (missing optional dep, dim mismatch, empty input)."""


class LabelError(KaosMLCoreError):
    """LLM labeling failure (no seeds, invalid class set, provider error)."""


class TrainError(KaosMLCoreError):
    """Classifier training failure (insufficient labels, single-class input)."""


class PredictError(KaosMLCoreError):
    """Prediction-time failure (feature shape mismatch, model not loaded)."""


__all__ = [
    "CorpusError",
    "FeatureError",
    "KaosMLCoreError",
    "LabelError",
    "PredictError",
    "TrainError",
]
