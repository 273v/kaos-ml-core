"""Shared helpers for the USC-based v0 acceptance tests.

These helpers exist so the three tests (pipeline integrity, LLM label
quality, end-to-end cold start) can share fixture loading, embedding,
and metric computation without duplicating code.

Every test that uses these helpers prints a human-readable report block
to stdout — the report blocks are the artifact a practicing lawyer
would actually read.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

from kaos_ml_core.cluster import (
    ClusterResult,
    kmedoid_seeds,
    minibatch_kmeans,
)
from kaos_ml_core.corpus import Corpus
from kaos_ml_core.features import embed_corpus
from tests.fixtures.usc_corpus import (
    LABEL_CRIMINAL,
    LABEL_TAX,
    TITLE_CRIMINAL,
    TITLE_TAX,
    TITLE_TO_LABEL,
    load_usc_corpus,
    split_train_eval,
)

# ── Tunables ────────────────────────────────────────────────────────────
# These are the values the three tests use. They're declared here so all
# three are reproducible from a single seed and so the report blocks are
# self-documenting.

PER_TITLE = 300  # 300 tax + 300 criminal = 600 total
EVAL_FRACTION = 0.33  # 198 held-out, 402 training pool
N_CLUSTERS = 20  # k-medoid clusters over the training pool
SEEDS_PER_CLUSTER = 3  # → 60 seeds total (max)
DEFAULT_RANDOM_STATE = 42

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "claude-haiku-4-5"

LABELS: list[str] = [LABEL_CRIMINAL, LABEL_TAX]

# Default title pair (the "easy" pair: tax vs criminal). Other tests
# can override by passing ``titles=(...)`` to ``load_corpus_and_features``.
DEFAULT_TITLES: tuple[int, ...] = (TITLE_CRIMINAL, TITLE_TAX)


# ── Typed result container ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Confusion-matrix-backed evaluation report for one classifier run."""

    accuracy: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion: dict[tuple[str, str], int]  # (true, pred) -> count
    n_eval: int
    majority_baseline: float

    def above_baseline(self) -> float:
        return self.accuracy - self.majority_baseline


# ── Loader / featurizer caching ─────────────────────────────────────────
# fastembed downloads + builds the ONNX session on first use. We cache
# the (Corpus, X) pair across tests in this process so we only pay the
# embedding cost once per pytest run.

_corpus_cache: dict[tuple, tuple] = {}


def load_corpus_and_features(
    *,
    titles: tuple[int, ...] = DEFAULT_TITLES,
    per_title: int = PER_TITLE,
    eval_fraction: float = EVAL_FRACTION,
    seed: int = DEFAULT_RANDOM_STATE,
):
    """Load USC docs, build train/eval Corpora, embed both.

    Returns:
        (train_corpus, X_train, train_gt,
         eval_corpus,  X_eval,  eval_gt, labels_for_pair)

    where ``train_gt`` / ``eval_gt`` are dict[row_index -> label_string]
    and ``labels_for_pair`` is the sorted list of label strings used by
    this title pair (handed back so callers don't need to thread it).

    Cached per (titles, per_title, eval_fraction, seed) so the embedding
    step runs once per pytest process per parameter combination.
    """
    pytest.importorskip("kaos_nlp_transformers")

    key = (titles, per_title, eval_fraction, seed)
    if key in _corpus_cache:
        return _corpus_cache[key]

    docs, gt_by_uri = load_usc_corpus(titles=titles, per_title=per_title, seed=seed)
    train_docs, eval_docs = split_train_eval(
        docs, gt_by_uri, eval_fraction=eval_fraction, seed=seed
    )

    train_corpus = Corpus.from_documents(train_docs, level="paragraph")
    eval_corpus = Corpus.from_documents(eval_docs, level="paragraph")

    X_train = embed_corpus(train_corpus, model=EMBED_MODEL)
    X_eval = embed_corpus(eval_corpus, model=EMBED_MODEL)

    # Per-row ground truth (a corpus row is a paragraph; the label
    # belongs to its parent doc, identified by doc_uri)
    train_gt = {u.row: gt_by_uri[u.doc_uri] for u in train_corpus}
    eval_gt = {u.row: gt_by_uri[u.doc_uri] for u in eval_corpus}

    labels_for_pair = sorted({TITLE_TO_LABEL[t] for t in titles})

    out = (
        train_corpus,
        X_train,
        train_gt,
        eval_corpus,
        X_eval,
        eval_gt,
        labels_for_pair,
    )
    _corpus_cache[key] = out
    return out


def cluster_and_seed(
    X_train, *, n_clusters: int = N_CLUSTERS, per_cluster: int = SEEDS_PER_CLUSTER
) -> tuple[ClusterResult, list[int]]:
    """Cluster the training feature matrix and pick k-medoid seed rows."""
    clusters = minibatch_kmeans(
        X_train,
        n_clusters=n_clusters,
        random_state=DEFAULT_RANDOM_STATE,
    )
    seeds = kmedoid_seeds(X_train, clusters, per_cluster=per_cluster)
    return clusters, seeds


# ── Document-level prediction (the corpus is paragraph-level, but the
#    ground truth is document-level — we vote across paragraphs to get
#    a single per-document prediction) ─────────────────────────────────


def predict_documents(
    corpus: Corpus,
    X: np.ndarray,
    clf,
    *,
    threshold: float = 0.5,
) -> dict[str, str]:
    """Vote paragraph predictions up to document predictions.

    Returns dict[doc_uri -> predicted_label]. Each document votes via
    majority of its paragraph-level predictions, ties broken by sum of
    positive-class scores. This is the right granularity for evaluation
    because the ground truth (USC Title) is per-document, not per-paragraph.
    """
    classes = list(clf.classes_)
    proba = clf.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    para_pred = np.array(classes)[pred_idx]

    # Aggregate to document level
    doc_votes: dict[str, Counter] = {}
    doc_score: dict[str, float] = {}
    for u in corpus:
        doc_uri = u.doc_uri
        doc_votes.setdefault(doc_uri, Counter())[str(para_pred[u.row])] += 1
        # Track score on the SECOND class for tie-breaking
        doc_score[doc_uri] = doc_score.get(doc_uri, 0.0) + float(proba[u.row, 1])

    doc_pred: dict[str, str] = {}
    for doc_uri, votes in doc_votes.items():
        top = votes.most_common()
        if len(top) == 1:
            doc_pred[doc_uri] = top[0][0]
            continue
        if top[0][1] != top[1][1]:
            doc_pred[doc_uri] = top[0][0]
            continue
        # Tie: defer to per-doc score
        doc_pred[doc_uri] = (
            classes[1] if doc_score[doc_uri] / max(votes.total(), 1) >= threshold else classes[0]
        )

    return doc_pred


def evaluate(
    doc_pred: dict[str, str],
    doc_gt: dict[str, str],
    labels: list[str],
) -> EvalReport:
    """Compute accuracy, per-class precision/recall/F1, and confusion."""
    if set(doc_pred.keys()) != set(doc_gt.keys()):
        missing = set(doc_gt.keys()) - set(doc_pred.keys())
        extra = set(doc_pred.keys()) - set(doc_gt.keys())
        msg = (
            f"prediction set mismatch: {len(missing)} missing, {len(extra)} extra. "
            "Fix: ensure predict_documents iterated the same corpus as the "
            "ground truth was built from."
        )
        raise AssertionError(msg)

    confusion: dict[tuple[str, str], int] = {(t, p): 0 for t in labels for p in labels}
    for doc_uri, true_label in doc_gt.items():
        pred = doc_pred[doc_uri]
        if (true_label, pred) not in confusion:
            confusion[(true_label, pred)] = 0
        confusion[(true_label, pred)] += 1

    n = sum(confusion.values())
    correct = sum(confusion[(L, L)] for L in labels)
    accuracy = correct / n if n else 0.0

    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    for L in labels:
        tp = confusion[(L, L)]
        fp = sum(confusion[(o, L)] for o in labels if o != L)
        fn = sum(confusion[(L, o)] for o in labels if o != L)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_precision[L] = precision
        per_class_recall[L] = recall
        per_class_f1[L] = f1

    # Majority-class baseline = the prevalence of the most common true label
    true_counts = Counter(doc_gt.values())
    majority = max(true_counts.values()) / n if n else 0.0

    return EvalReport(
        accuracy=accuracy,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        confusion=confusion,
        n_eval=n,
        majority_baseline=majority,
    )


# ── Pretty-print a report block ─────────────────────────────────────────


def print_report(
    title: str,
    report: EvalReport,
    *,
    labels: list[str],
    extras: dict[str, str] | None = None,
) -> None:
    """Print a human-readable evaluation block to stdout."""
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")
    if extras:
        for k, v in extras.items():
            print(f"  {k:<30s} {v}")
    print(f"\n  n_eval (documents):           {report.n_eval}")
    print(f"  accuracy:                     {report.accuracy:.3f}")
    print(f"  majority baseline:            {report.majority_baseline:.3f}")
    print(f"  delta over baseline:          {report.above_baseline():+.3f}")
    print()
    for L in labels:
        p = report.per_class_precision[L]
        r = report.per_class_recall[L]
        f = report.per_class_f1[L]
        print(f"  {L:<20s} precision={p:.3f}  recall={r:.3f}  f1={f:.3f}")
    print()
    print("  confusion matrix (rows=true, cols=pred):")
    header = "                  " + "  ".join(f"{L:>14s}" for L in labels)
    print(header)
    for true_label in labels:
        row = "  " + f"{true_label:<16s}"
        for pred_label in labels:
            row += f"  {report.confusion[(true_label, pred_label)]:>14d}"
        print(row)
    print()


__all__ = [
    "DEFAULT_RANDOM_STATE",
    "EMBED_MODEL",
    "EVAL_FRACTION",
    "LABELS",
    "LLM_MODEL",
    "N_CLUSTERS",
    "PER_TITLE",
    "SEEDS_PER_CLUSTER",
    "EvalReport",
    "cluster_and_seed",
    "evaluate",
    "load_corpus_and_features",
    "predict_documents",
    "print_report",
]
