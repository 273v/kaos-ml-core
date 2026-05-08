"""Trained-pipeline persistence — save once, ship to prod, load anywhere.

A ``Pipeline`` bundles the fitted classifier, the embedding-model id and
revision (so featurization is reproducible bit-exactly), the operating
threshold, the class labels, and the kaos-ml-core version into a single
on-disk artifact. Two methods of consequence: ``save(path)`` and
``load(path)``.

The artifact is a **directory** at ``path`` containing:

- ``manifest.json`` — JSON metadata. First key is the magic string
  ``_kaos_pipeline_format`` whose presence is the load-time gate
  (audit-01 KMC-101-style hardening; mirrors kaos-graph A2-#4 pickle
  magic-byte protection).
- ``classifier.joblib`` — sklearn classifier serialized via ``joblib``
  (NOT pickle directly; joblib has a documented format and refuses to
  load anything that doesn't match its envelope).

Loading refuses files that don't carry the magic string, manifests with
incompatible ``format_version``, or directories missing either piece.

Downstream use cases:

- **TAR**: a paralegal trains a responsiveness classifier in a Jupyter
  session, ``pipeline.save("/case-foo/responsive-v1.kaos")``, ships to
  the production agent that reads it back. No need to thread embed
  model id + classifier file + threshold + class names manually.
- **Contract analytics**: per-tenant arbitration-clause classifiers
  saved to a registry, loaded on demand by a deployed agent.
- **MCP agentic flow**: tool ``kaos-ml-train`` creates a pipeline_id in
  session memory; ``kaos-ml-save-pipeline`` persists it; a different
  session calls ``kaos-ml-load-pipeline`` to pick it up. Uses this
  format under the hood.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from kaos_ml_core.corpus import Corpus
from kaos_ml_core.errors import PredictError
from kaos_ml_core.metrics import Metrics

if TYPE_CHECKING:
    from kaos_content.model.tabular import TabularDocument


__all__ = ["Pipeline", "PipelineError"]


# Magic string that identifies a kaos-ml-core pipeline manifest. Embedded
# at the top of manifest.json so a quick file-content check rules out
# random JSON files. Bumped to a new value if the on-disk format ever
# changes incompatibly.
_PIPELINE_MAGIC = "kaos-ml-core/pipeline:v1"
_PIPELINE_FORMAT_VERSION = 1


class PipelineError(Exception):
    """Raised by :meth:`Pipeline.save` and :meth:`Pipeline.load` on I/O,
    format, or compatibility errors."""


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A trained classification pipeline ready to apply to new corpora.

    Carries everything needed to reproduce the prediction step from a
    fresh kaos-content ``ContentDocument`` (or a built ``Corpus``):

    1. Featurize via ``EmbeddingModel.load(embed_model_id)`` (the
       kaos-nlp-transformers registry pins the revision; we ALSO record
       it here so you'd notice a registry SHA change between train
       and predict).
    2. Score via ``classifier.predict_proba``.
    3. Threshold-classify into ``classes``.

    Attributes are immutable; construct a new ``Pipeline`` to retrain or
    re-tune. Save with :meth:`save`; load with :meth:`load`.
    """

    embed_model_id: str
    """HuggingFace Hub id of the embedding model (e.g.
    ``BAAI/bge-small-en-v1.5``). Used at predict time to re-load the
    same featurizer. Must be present in
    ``kaos_nlp_transformers.REGISTRY``."""

    embed_revision: str
    """Pinned commit SHA of the embedding model at training time. Loaded
    from the kaos-nlp-transformers registry. ``Pipeline.load`` warns if
    this differs from the registry's current pin (potential reproducibility
    drift)."""

    classifier: Any
    """The fitted sklearn classifier (typically
    ``sklearn.linear_model.LogisticRegression``). Must expose
    ``predict_proba`` and ``classes_``."""

    threshold: float
    """Operating threshold on the positive-class probability. Predictions
    where ``score >= threshold`` are positive. Typically tuned on a
    held-out control set via :func:`kaos_ml_core.tune_threshold`."""

    classes: tuple[str, ...]
    """Class labels in canonical order. The positive class for
    binary classification is ``classes[1]`` (sklearn convention)."""

    kaos_ml_core_version: str
    """kaos-ml-core version that produced this pipeline. Recorded for
    cross-version diagnostics — load() refuses on incompatible MAJOR
    version bumps."""

    train_metrics: Metrics | None = None
    """Test-set evaluation metrics from training. Optional but recommended
    so consumers can see the recall CI / precision the pipeline shipped
    with — without re-running evaluation."""

    notes: str = ""
    """Free-form provenance string (training corpus URI, partner-review
    notes, etc.). Persisted verbatim in the manifest."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Extension point for downstream tooling. Persisted as JSON; anything
    not JSON-serializable will raise at save time. Reserve top-level keys
    starting with ``_kaos_`` — those are kaos-ml-core internal."""

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def predict(
        self,
        corpus: Corpus,
        *,
        batch_size: int = 32,
    ) -> TabularDocument:
        """Featurize a corpus + classify + threshold in one call.

        Builds on top of :func:`kaos_ml_core.features.embed_corpus`
        (which delegates to kaos-nlp-transformers' ``EmbeddingModel``)
        and :func:`kaos_ml_core.predict.predict_corpus`. The output is a
        ``TabularDocument`` with one row per ``CorpusUnit``, carrying
        AST ``block_ref``, ``doc_uri``, ``page``, ``section_ref``,
        ``section_title``, ``predicted_label``, ``score``, and
        ``above_threshold`` per row.

        Args:
            corpus: Any kaos-ml-core ``Corpus``. The granularity used
                here SHOULD match the granularity used at training time
                (a sentence-trained classifier on a paragraph corpus
                will produce poor results — that's a user bug, not a
                framework bug).
            batch_size: Forwarded to ``embed_corpus``. Adjust for
                memory-constrained hosts.

        Returns:
            ``TabularDocument`` with one row per CorpusUnit.

        Raises:
            FeatureError: If kaos-nlp-transformers is not installed
                (``[transformers]`` extra missing).
            PredictError: On shape mismatch.
        """
        from kaos_ml_core.features import embed_corpus
        from kaos_ml_core.predict import predict_corpus

        x_features = embed_corpus(corpus, model=self.embed_model_id, batch_size=batch_size)
        positive_label = self.classes[1] if len(self.classes) >= 2 else None
        return predict_corpus(
            corpus,
            x_features,
            self.classifier,
            threshold=self.threshold,
            positive_label=positive_label,
        )

    def predict_proba(
        self,
        corpus: Corpus,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return raw positive-class probabilities for every row.

        Bypasses the threshold step. Useful for downstream re-tuning
        on a fresh control set without re-running featurization, or for
        producing a calibration plot.

        Returns:
            ``np.ndarray`` of shape ``(len(corpus),)``, dtype float64,
            values in [0, 1].
        """
        from kaos_ml_core.features import embed_corpus

        x_features = embed_corpus(corpus, model=self.embed_model_id, batch_size=batch_size)
        if not hasattr(self.classifier, "predict_proba"):
            msg = (
                f"Classifier {type(self.classifier).__name__} does not "
                "support predict_proba. Fix: use LogisticRegression or wrap "
                "in CalibratedClassifierCV before saving the pipeline."
            )
            raise PredictError(msg)
        proba = self.classifier.predict_proba(x_features)
        # Positive class column. classifier.classes_ ordering may differ
        # from self.classes — find by label match to be safe.
        clf_classes = list(self.classifier.classes_)
        positive_label = self.classes[1] if len(self.classes) >= 2 else self.classes[0]
        if positive_label not in clf_classes:
            msg = (
                f"Positive class {positive_label!r} not in classifier classes "
                f"{clf_classes}. Pipeline metadata and classifier disagree — "
                "the pipeline may have been edited or the classifier swapped."
            )
            raise PredictError(msg)
        pos_idx = clf_classes.index(positive_label)
        return np.asarray(proba[:, pos_idx], dtype=np.float64)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save the pipeline to ``path`` (a directory).

        Creates the directory if it doesn't exist. Writes:

        - ``manifest.json`` — magic + version + metadata + class names
        - ``classifier.joblib`` — sklearn classifier via joblib

        Args:
            path: Target directory. The ``.kaos`` suffix is conventional
                but not required.

        Returns:
            The resolved absolute ``Path`` written to.

        Raises:
            PipelineError: On I/O failure or non-JSON-serializable
                ``extras``.
        """
        target = Path(path).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)

        # Lazy-import joblib so callers without it can still construct +
        # apply Pipelines; persistence requires it.
        try:
            joblib = importlib.import_module("joblib")
        except ImportError as exc:
            msg = (
                "Pipeline.save requires joblib (transitive sklearn dep). "
                "Fix: install joblib or upgrade scikit-learn (which already "
                "depends on joblib)."
            )
            raise PipelineError(msg) from exc

        manifest: dict[str, Any] = {
            "_kaos_pipeline_format": _PIPELINE_MAGIC,
            "format_version": _PIPELINE_FORMAT_VERSION,
            "embed_model_id": self.embed_model_id,
            "embed_revision": self.embed_revision,
            "threshold": float(self.threshold),
            "classes": list(self.classes),
            "kaos_ml_core_version": self.kaos_ml_core_version,
            "train_metrics": _metrics_to_dict(self.train_metrics),
            "notes": self.notes,
            "extras": self.extras,
        }

        try:
            (target / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except (TypeError, OSError) as exc:
            msg = (
                f"Failed to serialize manifest to {target}/manifest.json: {exc}. "
                "Fix: verify Pipeline.extras is JSON-serializable (no numpy "
                "arrays, classes, or callables — convert to plain types first)."
            )
            raise PipelineError(msg) from exc

        try:
            joblib.dump(self.classifier, target / "classifier.joblib")
        except Exception as exc:
            msg = (
                f"Failed to dump classifier to {target}/classifier.joblib: "
                f"{exc}. Fix: ensure the classifier is a fitted sklearn "
                "estimator (joblib doesn't support arbitrary objects)."
            )
            raise PipelineError(msg) from exc

        return target

    @classmethod
    def load(cls, path: str | Path) -> Pipeline:
        """Load a pipeline saved via :meth:`save`.

        Validates:

        1. ``path`` is a directory.
        2. ``manifest.json`` exists and starts with the magic string.
        3. ``format_version`` is loadable by this kaos-ml-core build.
        4. ``classifier.joblib`` exists.

        On registry-revision drift (the embed model's pinned SHA in the
        live registry differs from what the manifest recorded), emits a
        ``RuntimeWarning`` but does NOT refuse — the user may have
        intentionally pinned an older revision.

        Args:
            path: Directory written by :meth:`save`.

        Returns:
            A ``Pipeline`` instance ready to ``predict()``.

        Raises:
            PipelineError: On any validation failure.
        """
        source = Path(path).expanduser().resolve()
        if not source.is_dir():
            msg = (
                f"Pipeline path {source} is not a directory. Fix: pass the "
                "path you used in Pipeline.save() (a directory), not a file."
            )
            raise PipelineError(msg)

        manifest_path = source / "manifest.json"
        classifier_path = source / "classifier.joblib"
        if not manifest_path.is_file():
            msg = (
                f"Manifest not found at {manifest_path}. Fix: this directory "
                "doesn't look like a kaos-ml-core pipeline. Did you pass the "
                "right path?"
            )
            raise PipelineError(msg)
        if not classifier_path.is_file():
            msg = (
                f"Classifier not found at {classifier_path}. Fix: directory "
                "is missing the classifier.joblib payload (corrupt or partial "
                "save?)."
            )
            raise PipelineError(msg)

        # Magic-byte gate — the manifest's first JSON key (sorted) is
        # ``_kaos_pipeline_format`` and its value is _PIPELINE_MAGIC. We
        # check both before trusting any other field, matching the
        # kaos-graph A2-#4 pickle hardening pattern.
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = (
                f"manifest.json is not valid JSON: {exc}. The pipeline "
                "directory is corrupt or was not produced by Pipeline.save."
            )
            raise PipelineError(msg) from exc

        if manifest.get("_kaos_pipeline_format") != _PIPELINE_MAGIC:
            msg = (
                f"manifest.json does not carry the kaos-ml-core pipeline "
                f"magic string {_PIPELINE_MAGIC!r}. Fix: this is not a "
                "kaos-ml-core pipeline directory; pass a path written by "
                "Pipeline.save()."
            )
            raise PipelineError(msg)

        format_version = manifest.get("format_version", 0)
        if format_version > _PIPELINE_FORMAT_VERSION:
            msg = (
                f"manifest format_version={format_version} is newer than "
                f"this kaos-ml-core build supports (max="
                f"{_PIPELINE_FORMAT_VERSION}). Fix: upgrade kaos-ml-core."
            )
            raise PipelineError(msg)

        try:
            joblib = importlib.import_module("joblib")
        except ImportError as exc:
            msg = (
                "Pipeline.load requires joblib (transitive sklearn dep). "
                "Fix: install joblib or upgrade scikit-learn."
            )
            raise PipelineError(msg) from exc

        try:
            classifier = joblib.load(classifier_path)
        except Exception as exc:
            msg = (
                f"Failed to load classifier from {classifier_path}: {exc}. "
                "Fix: the file may be corrupt or produced by an "
                "incompatible sklearn version. Verify the source environment."
            )
            raise PipelineError(msg) from exc

        # Optional registry-revision drift warning. We don't fail because
        # the user may have deliberately pinned an older revision via the
        # manifest's embed_revision; we just surface the drift.
        _check_revision_drift(
            manifest.get("embed_model_id", ""),
            manifest.get("embed_revision", ""),
        )

        return cls(
            embed_model_id=str(manifest.get("embed_model_id", "")),
            embed_revision=str(manifest.get("embed_revision", "")),
            classifier=classifier,
            threshold=float(manifest.get("threshold", 0.5)),
            classes=tuple(manifest.get("classes", ())),
            kaos_ml_core_version=str(manifest.get("kaos_ml_core_version", "")),
            train_metrics=_metrics_from_dict(manifest.get("train_metrics")),
            notes=str(manifest.get("notes", "")),
            extras=dict(manifest.get("extras", {})),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _metrics_to_dict(metrics: Metrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        # asdict() doesn't handle the tuple-of-tuples confusion_matrix
        # cleanly across re-loads; do it explicitly.
        **asdict(metrics),
        "confusion_matrix": [list(row) for row in metrics.confusion_matrix],
        "classes": list(metrics.classes),
    }


def _metrics_from_dict(payload: dict[str, Any] | None) -> Metrics | None:
    if not payload:
        return None
    return Metrics(
        precision=float(payload["precision"]),
        recall=float(payload["recall"]),
        f1=float(payload["f1"]),
        accuracy=float(payload["accuracy"]),
        support=int(payload["support"]),
        n_total=int(payload["n_total"]),
        recall_ci_lower=float(payload["recall_ci_lower"]),
        recall_ci_upper=float(payload["recall_ci_upper"]),
        confidence=float(payload["confidence"]),
        confusion_matrix=tuple(tuple(int(c) for c in row) for row in payload["confusion_matrix"]),
        classes=tuple(str(c) for c in payload["classes"]),
        threshold=payload.get("threshold"),
        roc_auc=payload.get("roc_auc"),
        extras=dict(payload.get("extras", {})),
    )


def _check_revision_drift(model_id: str, manifest_revision: str) -> None:
    """If kaos-nlp-transformers is installed, warn when the live registry's
    pinned revision differs from what the saved pipeline recorded."""
    if not model_id or not manifest_revision:
        return
    try:
        registry_module = importlib.import_module("kaos_nlp_transformers.models")
    except ImportError:
        return  # [transformers] extra not installed; nothing to compare.
    registry = getattr(registry_module, "REGISTRY", {})
    entry = registry.get(model_id)
    if entry is None:
        return  # Model not in the registry; user is on their own.
    live_revision = getattr(entry, "revision", None)
    if live_revision and live_revision != manifest_revision:
        import warnings

        warnings.warn(
            f"Pipeline manifest pins {model_id} @ {manifest_revision}, but "
            f"the installed kaos-nlp-transformers registry has it at "
            f"{live_revision}. Predictions will use the registry-pinned "
            "revision (which is what's actually on disk in the model cache); "
            "expect minor drift if the SHAs are different. To restore exact "
            "reproducibility, install the kaos-nlp-transformers version that "
            f"pinned {manifest_revision}.",
            RuntimeWarning,
            stacklevel=3,
        )
