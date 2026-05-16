"""MCP tool surface for kaos-ml-core — agentic access to the ML pipeline.

11 tools spanning the full classifier lifecycle:

  Build:      kaos-ml-build-corpus
  Inspect:    kaos-ml-corpus-info
  Cluster:    kaos-ml-cluster
  Label:      kaos-ml-label-seeds-with-llm
  Train:      kaos-ml-train
  Evaluate:   kaos-ml-evaluate
  Tune:       kaos-ml-tune-threshold
  Predict:    kaos-ml-predict
  Aggregate:  kaos-ml-aggregate
  Persist:    kaos-ml-save-pipeline, kaos-ml-load-pipeline

Builds on top of kaos-mcp's ``KaosTool`` ABC, ``ToolAnnotations``, and
``ParameterSchema``. Tool descriptions explicitly reference prerequisite
+ follow-up tools (per docs/guides/tool-design.md) so an agent can chain
them without external orchestration.

Session state lives in module-level registries keyed by
``KaosContext.session_id`` (matches the kaos-tabular pattern). Each
session sees only its own corpora / pipelines / predictions.

Downstream use cases this enables:

- A paralegal in Claude Code: "build a corpus from /case-foo/, cluster
  + label-seeds, train, evaluate, predict on the rest, give me the
  doc-level table." Eight tool calls; no Python required.
- A data-scientist agent: same, but the agent picks the granularity
  (paragraph for arbitration; document for responsiveness) based on
  the partner's natural-language ask.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from kaos_core.base.context import KaosContext
from kaos_core.base.tool import KaosTool
from kaos_core.registry.container import KaosRuntime
from kaos_core.types.annotations import ToolAnnotations
from kaos_core.types.enums import ToolCapability, ToolCategory
from kaos_core.types.metadata import ToolMetadata
from kaos_core.types.parameters import ParameterSchema
from kaos_core.types.results import ToolResult

from kaos_ml_core import __version__ as _VERSION
from kaos_ml_core.aggregate import aggregate_predictions
from kaos_ml_core.cluster import kmedoid_seeds, minibatch_kmeans
from kaos_ml_core.corpus import Corpus
from kaos_ml_core.metrics import evaluate
from kaos_ml_core.pipeline import Pipeline, PipelineError
from kaos_ml_core.predict import predict_corpus
from kaos_ml_core.split import stratified_split
from kaos_ml_core.threshold import tune_threshold
from kaos_ml_core.train import train_logreg

__all__ = ["register_ml_tools"]

_MODULE = "kaos-ml-core"


# ---------------------------------------------------------------------------
# Session-scoped registries (same pattern as kaos_tabular.tools._ENGINES)
# ---------------------------------------------------------------------------

# corpus_id → Corpus
_CORPORA: dict[str, dict[str, Corpus]] = {}
# pipeline_id → (Pipeline, feature_matrix_used_at_train_time)
# (we cache the feature matrix because evaluate / tune / predict can
# avoid re-embedding when the user is staying within one session.)
_PIPELINES: dict[str, dict[str, tuple[Pipeline, np.ndarray | None]]] = {}
# prediction_id → TabularDocument
_PREDICTIONS: dict[str, dict[str, Any]] = {}
# Per-session cluster results (k-medoid seed indices), keyed by corpus_id.
_CLUSTERS: dict[str, dict[str, dict[str, Any]]] = {}


def _session_id(context: KaosContext | None) -> str:
    """Resolve the session id, defaulting to a stable shared key when no
    context is provided (script mode)."""
    if context is None:
        return "__default__"
    return getattr(context, "session_id", None) or "__default__"


def _get_corpus(context: KaosContext | None, corpus_id: str) -> Corpus:
    sid = _session_id(context)
    bucket = _CORPORA.get(sid, {})
    if corpus_id not in bucket:
        msg = (
            f"corpus_id={corpus_id!r} not found in session. "
            "How to fix: call kaos-ml-build-corpus first to create one. "
            "Alternative: list known corpora via the session inspector."
        )
        raise KeyError(msg)
    return bucket[corpus_id]


def _put_corpus(context: KaosContext | None, corpus: Corpus) -> str:
    sid = _session_id(context)
    bucket = _CORPORA.setdefault(sid, {})
    corpus_id = f"corpus_{len(bucket) + 1}"
    bucket[corpus_id] = corpus
    return corpus_id


def _get_pipeline(
    context: KaosContext | None, pipeline_id: str
) -> tuple[Pipeline, np.ndarray | None]:
    sid = _session_id(context)
    bucket = _PIPELINES.get(sid, {})
    if pipeline_id not in bucket:
        msg = (
            f"pipeline_id={pipeline_id!r} not found in session. "
            "How to fix: call kaos-ml-train to create one, or "
            "kaos-ml-load-pipeline to load a saved pipeline. "
            "Alternative: list known pipelines via kaos-ml-load-pipeline."
        )
        raise KeyError(msg)
    return bucket[pipeline_id]


def _put_pipeline(
    context: KaosContext | None, pipeline: Pipeline, features: np.ndarray | None = None
) -> str:
    sid = _session_id(context)
    bucket = _PIPELINES.setdefault(sid, {})
    pipeline_id = f"pipeline_{len(bucket) + 1}"
    bucket[pipeline_id] = (pipeline, features)
    return pipeline_id


def _put_prediction(context: KaosContext | None, predictions: Any) -> str:
    sid = _session_id(context)
    bucket = _PREDICTIONS.setdefault(sid, {})
    prediction_id = f"predictions_{len(bucket) + 1}"
    bucket[prediction_id] = predictions
    return prediction_id


def _get_prediction(context: KaosContext | None, prediction_id: str):
    sid = _session_id(context)
    bucket = _PREDICTIONS.get(sid, {})
    if prediction_id not in bucket:
        msg = (
            f"prediction_id={prediction_id!r} not found in session. "
            "How to fix: call kaos-ml-predict first to create one."
        )
        raise KeyError(msg)
    return bucket[prediction_id]


# ---------------------------------------------------------------------------
# Build + inspect
# ---------------------------------------------------------------------------


class BuildCorpusTool(KaosTool):
    """Build an AST-grounded Corpus from previously-extracted ContentDocuments."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-build-corpus",
            display_name="Build Corpus",
            description=(
                "Build an AST-grounded Corpus from one or more ContentDocuments "
                "at the requested granularity (paragraph / sentence / section / "
                "document). Returns a corpus_id usable by every other kaos-ml-* "
                "tool. Prerequisites: extract documents first via kaos-pdf, "
                "kaos-office, kaos-web, or kaos-source."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=False,
            ),
            input_schema=[
                ParameterSchema(
                    name="documents_resource_id",
                    type="string",
                    description=(
                        "Resource id of a kaos-content ContentDocument tuple in "
                        "the session artifact store (e.g. from kaos-pdf-parse-pdf "
                        "output). For now the tool also accepts a list of dicts "
                        "(see corpus.py)."
                    ),
                ),
                ParameterSchema(
                    name="level",
                    type="string",
                    description=(
                        "Granularity: 'paragraph' (default; clause-level), "
                        "'sentence' (finest), 'section' (group by section_ref; "
                        "due-diligence), 'document' (one row per doc; ediscovery "
                        "responsiveness)."
                    ),
                    required=False,
                    constraints={"enum": ["paragraph", "sentence", "section", "document"]},
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        # The wire-level documents_resource_id contract is intentionally
        # thin in v0 — agents typically pass the in-memory list directly
        # via the artifact store. The full resource-handle plumbing lands
        # in 0.1.0a2 (see CHANGELOG).
        documents = inputs.get("_documents") or inputs.get("documents")
        if not documents:
            return ToolResult.create_error(
                "kaos-ml-build-corpus requires a documents iterable. "
                "How to fix: pass a list of kaos_content.ContentDocument instances "
                "via the 'documents' input (or '_documents' for direct injection). "
                "Alternative: extract via kaos-pdf-parse-pdf or kaos-office-parse-* first."
            )
        level = inputs.get("level", "paragraph")
        try:
            corpus = Corpus.from_documents(documents, level=level)
        except Exception as exc:
            return ToolResult.create_error(
                f"Failed to build corpus at level={level!r}: {exc}. "
                "How to fix: verify the documents are valid kaos_content.ContentDocument "
                "instances and have metadata.source.uri set. "
                "Alternative: pass doc_uris=[...] to override."
            )
        corpus_id = _put_corpus(context, corpus)
        return ToolResult.create_success(
            output={
                "corpus_id": corpus_id,
                "n_units": len(corpus),
                "level": level,
                "n_documents": len({u.doc_uri for u in corpus}),
                "summary": (
                    f"Built {corpus_id} with {len(corpus)} {level} units "
                    f"across {len({u.doc_uri for u in corpus})} documents. "
                    "Use kaos-ml-cluster for cold-start seed selection, or "
                    "kaos-ml-corpus-info for inspection."
                ),
            }
        )


class CorpusInfoTool(KaosTool):
    """Inspect an in-session Corpus (read-only)."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-corpus-info",
            display_name="Corpus Info",
            description=(
                "Return statistics about a built Corpus: row count, doc count, "
                "average text length, granularity. Read-only."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
            input_schema=[
                ParameterSchema(
                    name="corpus_id",
                    type="string",
                    description="ID returned by kaos-ml-build-corpus.",
                )
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            corpus = _get_corpus(context, inputs["corpus_id"])
        except KeyError as exc:
            return ToolResult.create_error(str(exc))
        n = len(corpus)
        total_chars = sum(len(corpus.unit(i).text) for i in range(n))
        return ToolResult.create_success(
            output={
                "corpus_id": inputs["corpus_id"],
                "n_units": n,
                "n_documents": len({corpus.unit(i).doc_uri for i in range(n)}),
                "avg_chars_per_unit": (total_chars / n) if n else 0,
                "first_unit_text": corpus.unit(0).text[:200] if n else "",
            }
        )


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


class ClusterTool(KaosTool):
    """MiniBatchKMeans clustering + k-medoid seed selection."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-cluster",
            display_name="Cluster Corpus",
            description=(
                "Embed the corpus (kaos-nlp-transformers) and cluster with "
                "MiniBatchKMeans, then pick k-medoid seed rows for cold-start "
                "labeling. Returns a feature matrix handle (cached for reuse) "
                "and seed_indices ready for kaos-ml-label-seeds-with-llm."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=True,
            ),
            input_schema=[
                ParameterSchema(
                    name="corpus_id", type="string", description="From kaos-ml-build-corpus."
                ),
                ParameterSchema(
                    name="n_clusters",
                    type="integer",
                    description="Number of clusters (typical: 10-50).",
                    required=False,
                ),
                ParameterSchema(
                    name="per_cluster",
                    type="integer",
                    description="K-medoid seeds per cluster (typical: 1-3).",
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            corpus = _get_corpus(context, inputs["corpus_id"])
        except KeyError as exc:
            return ToolResult.create_error(str(exc))
        n_clusters = int(inputs.get("n_clusters", 20))
        per_cluster = int(inputs.get("per_cluster", 2))
        try:
            from kaos_ml_core.features import embed_corpus

            X = embed_corpus(corpus)
            cluster_result = minibatch_kmeans(X, n_clusters=n_clusters)
            seed_idx = kmedoid_seeds(X, cluster_result, per_cluster=per_cluster)
        except Exception as exc:
            return ToolResult.create_error(
                f"Cluster failed: {exc}. "
                "How to fix: verify the [transformers] extra is installed "
                "(needed for embedding). "
                "Alternative: clusters can be skipped if you already have "
                "labels — go straight to kaos-ml-train."
            )
        sid = _session_id(context)
        seed_list = list(seed_idx)
        _CLUSTERS.setdefault(sid, {})[inputs["corpus_id"]] = {
            "X": X,
            "cluster_labels": cluster_result.labels.tolist(),
            "seed_indices": seed_list,
        }
        return ToolResult.create_success(
            output={
                "corpus_id": inputs["corpus_id"],
                "n_clusters": n_clusters,
                "n_seeds": len(seed_list),
                "seed_indices": seed_list,
                "summary": (
                    f"Clustered {len(corpus)} units into {n_clusters} clusters; "
                    f"selected {len(seed_list)} k-medoid seeds. "
                    "Next: kaos-ml-label-seeds-with-llm to label them, then kaos-ml-train."
                ),
            }
        )


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


class LabelSeedsTool(KaosTool):
    """LLM-driven cold-start labeling of seed rows."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-label-seeds-with-llm",
            display_name="Label Seeds with LLM",
            description=(
                "Send each seed row's text to an LLM and ask for a class label. "
                "Builds on kaos-llm-core's classify program. NOT read-only — "
                "calls a paid LLM endpoint."
            ),
            category=ToolCategory.AGENT,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=True,
            ),
            input_schema=[
                ParameterSchema(
                    name="corpus_id", type="string", description="From kaos-ml-build-corpus."
                ),
                ParameterSchema(
                    name="seed_indices",
                    type="array",
                    description="Indices of rows to label (typically from kaos-ml-cluster).",
                ),
                ParameterSchema(
                    name="classes",
                    type="array",
                    description="Class label strings (e.g. ['responsive', 'non_responsive']).",
                ),
                ParameterSchema(
                    name="instructions",
                    type="string",
                    description="Per-row prompt explaining the classification task.",
                ),
                ParameterSchema(
                    name="model",
                    type="string",
                    description="LLM model id (e.g. claude-haiku-4-5). Required.",
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            corpus = _get_corpus(context, inputs["corpus_id"])
        except KeyError as exc:
            return ToolResult.create_error(str(exc))
        seed_idx = [int(i) for i in inputs["seed_indices"]]
        try:
            from kaos_ml_core.label import label_seeds_with_llm

            labels = await label_seeds_with_llm(
                corpus=corpus,
                seed_rows=seed_idx,
                classes=list(inputs["classes"]),
                instructions=inputs["instructions"],
                model=inputs["model"],
            )
        except Exception as exc:
            return ToolResult.create_error(
                f"LLM labeling failed: {exc}. "
                "How to fix: verify ANTHROPIC_API_KEY (or the relevant provider) "
                "is set, and the [llm] extra is installed. "
                "Alternative: provide labels manually as a list and skip this tool."
            )
        return ToolResult.create_success(
            output={
                "corpus_id": inputs["corpus_id"],
                "n_seeds": len(seed_idx),
                "labels": dict(labels),
                "summary": f"Labeled {len(labels)} seeds via LLM. Next: kaos-ml-train.",
            }
        )


# ---------------------------------------------------------------------------
# Train + evaluate + tune
# ---------------------------------------------------------------------------


class TrainTool(KaosTool):
    """Train a logistic-regression classifier on the labeled seeds."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-train",
            display_name="Train Classifier",
            description=(
                "Train a sklearn LogisticRegression(class_weight='balanced', "
                "solver='liblinear') on labeled rows of the corpus. Builds the "
                "feature matrix once (cached) and returns a pipeline_id usable "
                "by kaos-ml-evaluate / kaos-ml-tune-threshold / kaos-ml-predict / "
                "kaos-ml-save-pipeline."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=False,
            ),
            input_schema=[
                ParameterSchema(
                    name="corpus_id", type="string", description="From kaos-ml-build-corpus."
                ),
                ParameterSchema(
                    name="labels",
                    type="array",
                    description="Class label per row of the corpus, length == n_units.",
                ),
                ParameterSchema(
                    name="test_frac",
                    type="number",
                    description="Stratified test fraction (default 0.2).",
                    required=False,
                ),
                ParameterSchema(
                    name="control_frac",
                    type="number",
                    description="Held-out control fraction for threshold tuning (default 0.1).",
                    required=False,
                ),
                ParameterSchema(
                    name="seed",
                    type="integer",
                    description="Random seed (default 42).",
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            corpus = _get_corpus(context, inputs["corpus_id"])
        except KeyError as exc:
            return ToolResult.create_error(str(exc))
        labels_in = inputs["labels"]
        labels = np.asarray(labels_in)
        if labels.size != len(corpus):
            return ToolResult.create_error(
                f"labels length ({labels.size}) must match corpus size ({len(corpus)}). "
                "How to fix: pass one label per CorpusUnit, including for un-seed-labeled rows. "
                "Alternative: only have seed labels? Use a sentinel (e.g. ' ') for unknown rows "
                "and filter externally before kaos-ml-train."
            )
        try:
            from kaos_ml_core.features import embed_corpus

            X = embed_corpus(corpus)
            split = stratified_split(
                labels,
                test_frac=float(inputs.get("test_frac", 0.2)),
                control_frac=float(inputs.get("control_frac", 0.1)),
                seed=int(inputs.get("seed", 42)),
            )
            clf = train_logreg(X[split.train_idx], labels[split.train_idx])
            test_pred = clf.predict(X[split.test_idx])
            metrics = evaluate(
                labels[split.test_idx], test_pred, classes=tuple(np.unique(labels).tolist())
            )
        except Exception as exc:
            return ToolResult.create_error(
                f"Training failed: {exc}. "
                "How to fix: verify the corpus has been built, [transformers] extra "
                "is installed, and labels match corpus length. "
                "Alternative: pre-train with kaos_ml_core.train.train_logreg directly "
                "and call kaos-ml-load-pipeline to register."
            )
        # Look up the embed model's revision for reproducibility.
        embed_revision = ""
        try:
            from kaos_nlp_transformers import REGISTRY

            entry = REGISTRY.get("BAAI/bge-small-en-v1.5")
            embed_revision = entry.revision if entry else ""
        except ImportError:
            pass
        pipeline = Pipeline(
            embed_model_id="BAAI/bge-small-en-v1.5",
            embed_revision=embed_revision,
            classifier=clf,
            threshold=0.5,
            classes=tuple(str(c) for c in np.unique(labels).tolist()),
            kaos_ml_core_version=_VERSION,
            train_metrics=metrics,
        )
        pipeline_id = _put_pipeline(context, pipeline, X)
        return ToolResult.create_success(
            output={
                "pipeline_id": pipeline_id,
                "test_metrics": {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "recall_ci": [metrics.recall_ci_lower, metrics.recall_ci_upper],
                },
                "split": {
                    "n_train": split.n_train,
                    "n_test": split.n_test,
                    "n_control": split.n_control,
                    "control_idx": split.control_idx.tolist(),
                },
                "summary": (
                    f"Trained {pipeline_id}: F1={metrics.f1:.3f} "
                    f"recall={metrics.recall:.3f} (95% CI [{metrics.recall_ci_lower:.3f}, "
                    f"{metrics.recall_ci_upper:.3f}]). "
                    "Next: kaos-ml-tune-threshold on the control set, then kaos-ml-predict."
                ),
            }
        )


class TuneThresholdTool(KaosTool):
    """Tune the operating threshold on a held-out control set."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-tune-threshold",
            display_name="Tune Threshold",
            description=(
                "Find the operating threshold that hits target_recall (or "
                "target_precision) on a held-out control set. Updates the "
                "pipeline in place. Enforces CLAUDE.md hard rule #5 (no "
                "tuning on training data) by requiring the control_indices "
                "explicitly."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
            input_schema=[
                ParameterSchema(
                    name="pipeline_id", type="string", description="From kaos-ml-train."
                ),
                ParameterSchema(
                    name="corpus_id", type="string", description="From kaos-ml-build-corpus."
                ),
                ParameterSchema(
                    name="control_indices",
                    type="array",
                    description="Held-out control row indices (returned by kaos-ml-train).",
                ),
                ParameterSchema(
                    name="labels",
                    type="array",
                    description="Ground-truth labels (full-corpus length, same as kaos-ml-train).",
                ),
                ParameterSchema(
                    name="target_recall",
                    type="number",
                    description="Minimum recall to achieve (e.g. 0.85).",
                    required=False,
                ),
                ParameterSchema(
                    name="target_precision",
                    type="number",
                    description=(
                        "Minimum precision to achieve. Mutually exclusive with target_recall."
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            pipeline, X = _get_pipeline(context, inputs["pipeline_id"])
        except KeyError as exc:
            return ToolResult.create_error(str(exc))
        if X is None:
            return ToolResult.create_error(
                "pipeline has no cached feature matrix; threshold tuning needs "
                "the same X used for training. How to fix: re-run kaos-ml-train, "
                "or pre-compute X via kaos-ml-cluster (which caches it)."
            )
        labels = np.asarray(inputs["labels"])
        control_idx = np.asarray(inputs["control_indices"], dtype=int)
        try:
            proba = pipeline.classifier.predict_proba(X[control_idx])
            classes = list(pipeline.classifier.classes_)
            pos_label = pipeline.classes[1] if len(pipeline.classes) >= 2 else classes[-1]
            pos_idx = classes.index(pos_label)
            res = tune_threshold(
                labels[control_idx],
                proba[:, pos_idx],
                target_recall=inputs.get("target_recall"),
                target_precision=inputs.get("target_precision"),
                pos_label=pos_label,
            )
        except (ValueError, KeyError) as exc:
            return ToolResult.create_error(
                f"Tune threshold failed: {exc}. "
                "How to fix: pass exactly one of target_recall / target_precision, "
                "and verify control_indices comes from kaos-ml-train."
            )
        # Replace pipeline with new threshold (frozen dataclass — make a new one).
        from dataclasses import replace

        updated = replace(pipeline, threshold=res.threshold)
        sid = _session_id(context)
        _PIPELINES[sid][inputs["pipeline_id"]] = (updated, X)
        return ToolResult.create_success(
            output={
                "pipeline_id": inputs["pipeline_id"],
                "threshold": res.threshold,
                "achieved_recall": res.achieved_recall,
                "achieved_precision": res.achieved_precision,
                "n_control": res.n_control,
                "summary": (
                    f"Tuned threshold to {res.threshold:.3f} on control set "
                    f"(recall={res.achieved_recall:.3f}, "
                    f"precision={res.achieved_precision:.3f}). "
                    "Next: kaos-ml-predict to apply to the corpus."
                ),
            }
        )


class EvaluateTool(KaosTool):
    """Evaluate a pipeline on a held-out test set (read-only)."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-evaluate",
            display_name="Evaluate Pipeline",
            description=(
                "Evaluate a trained pipeline on a held-out subset of the corpus. "
                "Returns precision, recall, F1, accuracy, ROC AUC, confusion "
                "matrix, and a Wilson 95% CI on recall (CLAUDE.md hard rule #3). "
                "Read-only."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
            input_schema=[
                ParameterSchema(
                    name="pipeline_id", type="string", description="From kaos-ml-train."
                ),
                ParameterSchema(
                    name="row_indices",
                    type="array",
                    description=(
                        "Indices of rows to evaluate against (typically split.test_idx from "
                        "kaos-ml-train)."
                    ),
                ),
                ParameterSchema(
                    name="labels",
                    type="array",
                    description="Ground-truth labels (full-corpus length).",
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            pipeline, X = _get_pipeline(context, inputs["pipeline_id"])
        except KeyError as exc:
            return ToolResult.create_error(str(exc))
        if X is None:
            return ToolResult.create_error(
                "pipeline has no cached feature matrix; can't evaluate without X. "
                "How to fix: re-run kaos-ml-train, or pre-compute X via kaos-ml-cluster."
            )
        labels = np.asarray(inputs["labels"])
        idx = np.asarray(inputs["row_indices"], dtype=int)
        y_pred = pipeline.classifier.predict(X[idx])
        classes = list(pipeline.classifier.classes_)
        pos_label = pipeline.classes[1] if len(pipeline.classes) >= 2 else classes[-1]
        pos_idx = classes.index(pos_label)
        proba = pipeline.classifier.predict_proba(X[idx])[:, pos_idx]
        m = evaluate(labels[idx], y_pred, y_proba=proba, classes=pipeline.classes)
        return ToolResult.create_success(
            output={
                "pipeline_id": inputs["pipeline_id"],
                "n_evaluated": int(idx.size),
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "accuracy": m.accuracy,
                "support": m.support,
                "recall_ci": [m.recall_ci_lower, m.recall_ci_upper],
                "confidence": m.confidence,
                "roc_auc": m.roc_auc,
                "confusion_matrix": [list(r) for r in m.confusion_matrix],
                "summary": (
                    f"Evaluated on {idx.size} rows: F1={m.f1:.3f}, "
                    f"recall={m.recall:.3f} (Wilson 95% CI: [{m.recall_ci_lower:.3f}, "
                    f"{m.recall_ci_upper:.3f}])."
                ),
            }
        )


# ---------------------------------------------------------------------------
# Predict + aggregate
# ---------------------------------------------------------------------------


class PredictTool(KaosTool):
    """Apply a pipeline to a corpus → TabularDocument predictions."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-predict",
            display_name="Predict on Corpus",
            description=(
                "Apply a trained pipeline to every row of a corpus. Returns a "
                "predictions resource (TabularDocument with row, block_ref, "
                "doc_uri, score, predicted_label, above_threshold). Pass the "
                "predictions resource_id to kaos-ml-aggregate for "
                "cross-granularity decisions, or to kaos-tabular-query for SQL."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.QUERY,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
            input_schema=[
                ParameterSchema(
                    name="pipeline_id",
                    type="string",
                    description="From kaos-ml-train or kaos-ml-load-pipeline.",
                ),
                ParameterSchema(
                    name="corpus_id", type="string", description="From kaos-ml-build-corpus."
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            pipeline, X = _get_pipeline(context, inputs["pipeline_id"])
            corpus = _get_corpus(context, inputs["corpus_id"])
        except KeyError as exc:
            return ToolResult.create_error(str(exc))
        try:
            if X is not None and X.shape[0] == len(corpus):
                # In-session reuse: skip re-embedding.
                positive_label = pipeline.classes[1] if len(pipeline.classes) >= 2 else None
                predictions = predict_corpus(
                    corpus,
                    X,
                    pipeline.classifier,
                    threshold=pipeline.threshold,
                    positive_label=positive_label,
                )
            else:
                predictions = pipeline.predict(corpus)
        except Exception as exc:
            return ToolResult.create_error(
                f"Predict failed: {exc}. "
                "How to fix: verify [transformers] extra is installed and the "
                "corpus is non-empty. Alternative: kaos-ml-evaluate to confirm "
                "the pipeline is well-formed."
            )
        prediction_id = _put_prediction(context, predictions)
        n_rows = predictions.tables[0].row_count if predictions.tables else 0
        return ToolResult.create_success(
            output={
                "prediction_id": prediction_id,
                "pipeline_id": inputs["pipeline_id"],
                "corpus_id": inputs["corpus_id"],
                "n_rows": n_rows,
                "summary": (
                    f"Predicted on {n_rows} rows. Next: kaos-ml-aggregate to roll "
                    "up to doc_uri / section_ref, or kaos-tabular-query for SQL."
                ),
            }
        )


class AggregateTool(KaosTool):
    """Aggregate fine-grained predictions to a coarser key."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-aggregate",
            display_name="Aggregate Predictions",
            description=(
                "Aggregate row-level predictions to a coarser grouping key "
                "(typically doc_uri or section_ref) using one of: any (default; "
                "doc is positive if any row is), all, max, mean, count, "
                "majority. Returns a new TabularDocument; original predictions "
                "are untouched."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
            input_schema=[
                ParameterSchema(
                    name="prediction_id", type="string", description="From kaos-ml-predict."
                ),
                ParameterSchema(
                    name="by",
                    type="string",
                    description="Aggregation key column (doc_uri / section_ref / page).",
                    required=False,
                ),
                ParameterSchema(
                    name="method",
                    type="string",
                    description="Aggregation method.",
                    required=False,
                    constraints={"enum": ["any", "all", "max", "mean", "count", "majority"]},
                ),
                ParameterSchema(
                    name="positive_class",
                    type="string",
                    description=(
                        "Class label considered positive (default: most-frequent "
                        "above_threshold class)."
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            predictions = _get_prediction(context, inputs["prediction_id"])
        except KeyError as exc:
            return ToolResult.create_error(str(exc))
        try:
            agg = aggregate_predictions(
                predictions,
                by=inputs.get("by", "doc_uri"),
                method=inputs.get("method", "any"),
                positive_class=inputs.get("positive_class"),
            )
        except ValueError as exc:
            return ToolResult.create_error(
                f"Aggregate failed: {exc}. "
                "How to fix: verify the predictions table has the expected columns "
                "(produced by kaos-ml-predict)."
            )
        agg_id = _put_prediction(context, agg)
        return ToolResult.create_success(
            output={
                "aggregated_id": agg_id,
                "n_rows": agg.tables[0].row_count if agg.tables else 0,
                "by": inputs.get("by", "doc_uri"),
                "method": inputs.get("method", "any"),
                "summary": (
                    f"Aggregated by {inputs.get('by', 'doc_uri')!r} using "
                    f"method={inputs.get('method', 'any')!r}. "
                    "Use kaos-tabular-query to filter / sort the result."
                ),
            }
        )


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------


class SavePipelineTool(KaosTool):
    """Persist a pipeline to disk for cross-session reuse."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-save-pipeline",
            display_name="Save Pipeline",
            description=(
                "Save a trained pipeline to a directory containing "
                "manifest.json + classifier.joblib. Magic-byte hardened on "
                "load. Use to ship trained pipelines across sessions / hosts / "
                "deployments."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=True,
                openWorldHint=True,
            ),
            input_schema=[
                ParameterSchema(
                    name="pipeline_id", type="string", description="From kaos-ml-train."
                ),
                ParameterSchema(
                    name="path", type="string", description="Directory to save the pipeline to."
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            pipeline, _ = _get_pipeline(context, inputs["pipeline_id"])
        except KeyError as exc:
            return ToolResult.create_error(str(exc))
        try:
            saved = pipeline.save(inputs["path"])
        except PipelineError as exc:
            return ToolResult.create_error(
                f"Save failed: {exc}. "
                "How to fix: verify the path is writable and Pipeline.extras is "
                "JSON-serializable. Alternative: pass a different path."
            )
        return ToolResult.create_success(
            output={
                "pipeline_id": inputs["pipeline_id"],
                "path": str(saved),
                "summary": f"Saved pipeline to {saved}.",
            }
        )


class LoadPipelineTool(KaosTool):
    """Load a previously-saved pipeline from disk."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="kaos-ml-load-pipeline",
            display_name="Load Pipeline",
            description=(
                "Load a pipeline saved via kaos-ml-save-pipeline. Returns a "
                "pipeline_id usable by kaos-ml-predict / kaos-ml-evaluate. "
                "Validates the magic-byte header and refuses files that don't "
                "carry it."
            ),
            category=ToolCategory.DATA,
            capability=ToolCapability.TRANSFORM,
            module_name=_MODULE,
            version=_VERSION,
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=True,
            ),
            input_schema=[
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Directory written by kaos-ml-save-pipeline.",
                ),
            ],
        )

    async def execute(
        self, inputs: dict[str, Any], context: KaosContext | None = None
    ) -> ToolResult:
        try:
            pipeline = Pipeline.load(inputs["path"])
        except PipelineError as exc:
            return ToolResult.create_error(
                f"Load failed: {exc}. "
                "How to fix: verify the path was written by kaos-ml-save-pipeline "
                "and is intact."
            )
        # Without an in-session feature matrix, predict() will re-embed
        # the corpus on each call. That's fine — just slower than reusing
        # an in-session X.
        pipeline_id = _put_pipeline(context, pipeline, features=None)
        return ToolResult.create_success(
            output={
                "pipeline_id": pipeline_id,
                "embed_model_id": pipeline.embed_model_id,
                "embed_revision": pipeline.embed_revision,
                "classes": list(pipeline.classes),
                "threshold": pipeline.threshold,
                "summary": (
                    f"Loaded {pipeline_id} (model={pipeline.embed_model_id}, "
                    f"threshold={pipeline.threshold:.3f}). "
                    "Next: kaos-ml-predict on a corpus."
                ),
            }
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_ml_tools(runtime: KaosRuntime) -> int:
    """Register every kaos-ml-core MCP tool on the given runtime.

    Returns the number of tools registered (currently 11). Subsequent
    releases may add tools (active learning, calibration, batch
    evaluation); the count grows monotonically until a 1.0 tag fixes
    the surface.
    """
    tools: list[KaosTool] = [
        BuildCorpusTool(),
        CorpusInfoTool(),
        ClusterTool(),
        LabelSeedsTool(),
        TrainTool(),
        EvaluateTool(),
        TuneThresholdTool(),
        PredictTool(),
        AggregateTool(),
        SavePipelineTool(),
        LoadPipelineTool(),
    ]
    for tool in tools:
        runtime.tools.register_tool(tool)
    return len(tools)
