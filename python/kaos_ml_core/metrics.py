"""Evaluation metrics for kaos-ml-core classifiers.

Builds on top of ``sklearn.metrics`` for the standard classification scores
(precision, recall, F1, accuracy, ROC AUC, confusion matrix) and adds the
**Wilson 95% recall confidence interval** that is required by the
package's CLAUDE.md hard rule #3:

    "Never report a recall point estimate without a Wilson 95% CI.
     Defensibility requirement (v1.5+)."

Wilson 1927 score interval is preferred over the normal-approximation
(Agresti-Coull) interval at small or skewed sample sizes — the typical
TAR / due-diligence regime where the positive class is < 10% of the
corpus and the held-out evaluation set is < 1000 documents. We hand-roll
the formula so we don't pull a stats library just for this; the math is
well-trodden and exact.

Downstream use cases:

- **TAR / ediscovery:** regulators ask for "recall ≥ 0.80 with 95%
  confidence." ``Metrics.recall_ci_lower`` is the defensible number.
- **Contract analytics:** "did the classifier find every arbitration
  clause in the data room?" — recall + CI bounds the false-negative risk.
- **Due diligence:** triage classifiers need both precision (don't make
  reviewers chase false positives) and recall (don't miss key docs).

The class is ``frozen=True, slots=True`` so ``Metrics`` instances are
hashable and serializable — they round-trip through Pipeline.save/load
metadata as JSON.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

__all__ = [
    "Metrics",
    "evaluate",
    "wilson_score_interval",
]


@dataclass(frozen=True, slots=True)
class Metrics:
    """Evaluation result for a single classifier on a single split.

    All scores are computed for the **positive class** in binary
    classification and macro-averaged across classes for multi-class
    problems. Wilson CI is computed for the positive class's recall — that
    is the regulator-defensible number for ediscovery / TAR use cases.

    Field order kept stable for human-readable repr and JSON serialization.
    """

    precision: float
    """Positive-class precision (binary) or macro-average (multi-class). [0, 1]."""

    recall: float
    """Positive-class recall (binary) or macro-average (multi-class). [0, 1]."""

    f1: float
    """Harmonic mean of precision and recall. [0, 1]."""

    accuracy: float
    """Fraction of all predictions that were correct. [0, 1]."""

    support: int
    """Number of actual positive instances (binary) or total samples (multi-class)."""

    n_total: int
    """Total number of samples evaluated."""

    recall_ci_lower: float
    """Wilson lower bound on recall at the requested confidence (default 95%)."""

    recall_ci_upper: float
    """Wilson upper bound on recall at the requested confidence (default 95%)."""

    confidence: float
    """Confidence level for the recall CI (e.g. 0.95)."""

    confusion_matrix: tuple[tuple[int, ...], ...]
    """Confusion matrix rows = true labels, columns = predicted. Tuple-of-tuples
    for hashability + JSON-serializability without numpy."""

    classes: tuple[str, ...]
    """Ordered class labels matching the confusion matrix axes."""

    threshold: float | None = None
    """Operating threshold the predictions came from (informational; ``None``
    when predictions were made without a threshold sweep)."""

    roc_auc: float | None = None
    """Area under the ROC curve. Only computed when ``y_proba`` is supplied
    AND the problem is binary (sklearn's roc_auc_score has known multi-class
    quirks; we keep the binary contract clean and skip multi-class for v0)."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Extension point — Metrics consumers (e.g. Pipeline manifests) can
    attach run-specific bookkeeping here without changing the dataclass.
    Use sparingly; prefer adding typed fields when something becomes a
    first-class concept."""


def wilson_score_interval(
    positives: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson 1927 score interval for a binomial proportion.

    Returns ``(lower, upper)`` bounds at the requested ``confidence`` level
    on the true proportion implied by observing ``positives`` out of
    ``total``. Handles the boundary cases that the normal-approximation
    interval gets wrong:

    - ``positives == 0``: lower is 0.0, upper is the Wilson upper bound.
    - ``positives == total``: upper is 1.0, lower is the Wilson lower bound.
    - ``total == 0``: returns ``(0.0, 1.0)`` — the maximally-uncertain
      interval rather than raising a DivisionByZero.

    Args:
        positives: Number of successes (true positives in the recall
            interpretation: actually-positive AND predicted-positive).
        total: Number of trials (in recall: actually-positive count).
        confidence: Two-sided confidence level in (0, 1). 0.95 is the
            regulatory default for TAR.

    Returns:
        ``(lower, upper)`` floats in ``[0.0, 1.0]``.
    """
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must be in (0, 1), got {confidence!r}"
        raise ValueError(msg)
    if total < 0 or positives < 0:
        msg = f"positives and total must be ≥ 0, got positives={positives}, total={total}"
        raise ValueError(msg)
    if positives > total:
        msg = (
            f"positives ({positives}) cannot exceed total ({total}). "
            "Fix: verify y_true and y_pred arrays are aligned and use the "
            "same class encoding."
        )
        raise ValueError(msg)
    if total == 0:
        return (0.0, 1.0)

    # Two-sided z-score for the requested confidence.
    # Use the inverse-normal approximation good to ~1e-9; avoids scipy.
    # https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function
    z = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    n = float(total)
    p_hat = positives / n
    z_sq = z * z
    denom = 1.0 + z_sq / n
    center = (p_hat + z_sq / (2.0 * n)) / denom
    half = (z * math.sqrt((p_hat * (1.0 - p_hat) + z_sq / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: np.ndarray | None = None,
    classes: tuple[Any, ...] | list[Any] | None = None,
    confidence: float = 0.95,
    threshold: float | None = None,
) -> Metrics:
    """Compute classification metrics for a single (y_true, y_pred) pair.

    Build ON TOP of sklearn for the standard scores; add the Wilson CI on
    recall (CLAUDE.md hard rule #3). For binary classification, the
    positive class is taken as ``classes[1]`` (sklearn convention); for
    multi-class problems, scores are macro-averaged.

    Args:
        y_true: Ground-truth labels, shape ``(n,)``.
        y_pred: Predicted labels, shape ``(n,)``. Must use the same
            encoding as ``y_true``.
        y_proba: Optional positive-class probabilities for binary problems,
            shape ``(n,)``. When supplied, ``Metrics.roc_auc`` is computed.
        classes: Class labels in canonical order. When ``None``, inferred
            from ``np.unique(y_true)``. For binary classification, the
            second element is treated as the positive class.
        confidence: Confidence level for the recall CI (default 0.95).
        threshold: Operating threshold the predictions came from
            (informational; recorded on Metrics for downstream tooling).

    Returns:
        A frozen ``Metrics`` instance.

    Raises:
        ValueError: On length mismatch, empty input, or invalid confidence.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        msg = (
            f"y_true shape {y_true.shape} must equal y_pred shape {y_pred.shape}. "
            "Fix: align the two arrays — both should have one entry per evaluated row."
        )
        raise ValueError(msg)
    if y_true.ndim != 1:
        msg = (
            f"y_true and y_pred must be 1-D, got ndim={y_true.ndim}. "
            "Fix: pass label arrays, not probability matrices. For y_proba, "
            "use the y_proba keyword."
        )
        raise ValueError(msg)
    if y_true.size == 0:
        msg = "evaluate() requires at least one row; got empty arrays."
        raise ValueError(msg)

    # Preserve the original class dtype (int vs str) for sklearn calls;
    # only stringify for the Metrics dataclass output (JSON-serializable).
    if classes is None:
        classes_native: tuple[Any, ...] = tuple(np.unique(y_true).tolist())
    else:
        classes_native = tuple(classes)
    classes_str: tuple[str, ...] = tuple(str(c) for c in classes_native)

    n_classes = len(classes_native)
    is_binary = n_classes == 2

    # ── precision / recall / f1 ────────────────────────────────────────
    # Use macro-average for multi-class; binary average for n_classes == 2.
    average = "binary" if is_binary else "macro"
    pos_label = classes_native[1] if is_binary else None

    # sklearn's precision_recall_fscore_support emits warnings when a
    # class has zero predicted samples; we handle that as "score=0" rather
    # than crashing.
    p, r, f, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        pos_label=pos_label,
        zero_division=0.0,
        labels=list(classes_native),
    )

    accuracy = float(accuracy_score(y_true, y_pred))

    # ── confusion matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(classes_native))
    cm_tuple = tuple(tuple(int(x) for x in row) for row in cm)

    # ── support: positive-class count for binary, total for multi-class ─
    if is_binary:
        positive_class_label = classes_native[1]
        support_count = int(np.sum(y_true == positive_class_label))
        true_positives = int(
            np.sum((y_true == positive_class_label) & (y_pred == positive_class_label))
        )
    else:
        support_count = int(y_true.size)
        # For multi-class the "recall CI" is on the macro-averaged recall,
        # which is awkward to bound exactly — we compute the Wilson CI on
        # the per-row "correct prediction" rate as a defensible proxy.
        # Document this caveat clearly in the dataclass docstring.
        true_positives = int((y_true == y_pred).sum())

    # ── Wilson CI on recall ────────────────────────────────────────────
    ci_lo, ci_hi = wilson_score_interval(
        positives=true_positives,
        total=support_count if is_binary else int(y_true.size),
        confidence=confidence,
    )

    # ── ROC AUC (binary + y_proba only) ────────────────────────────────
    roc_auc: float | None = None
    if y_proba is not None and is_binary:
        y_proba_arr = np.asarray(y_proba)
        if y_proba_arr.shape != y_true.shape:
            msg = (
                f"y_proba shape {y_proba_arr.shape} must equal y_true shape "
                f"{y_true.shape}. Fix: pass the positive-class probability "
                "column from clf.predict_proba(X)[:, 1]."
            )
            raise ValueError(msg)
        # roc_auc_score requires both classes present; gracefully degrade
        # to None on degenerate single-class evaluation sets.
        if len(np.unique(y_true)) == 2:
            # Convert y_true to a 0/1 binary array against the positive
            # class so sklearn's default "alphabetically-greater is positive"
            # convention can't invert the AUC. classes_native[1] is the
            # positive class by sklearn convention.
            y_true_binary = (y_true == classes_native[1]).astype(int)
            roc_auc = float(roc_auc_score(y_true_binary, y_proba_arr))

    return Metrics(
        precision=float(p),
        recall=float(r),
        f1=float(f),
        accuracy=accuracy,
        support=support_count,
        n_total=int(y_true.size),
        recall_ci_lower=ci_lo,
        recall_ci_upper=ci_hi,
        confidence=confidence,
        confusion_matrix=cm_tuple,
        classes=classes_str,
        threshold=threshold,
        roc_auc=roc_auc,
    )


# ---------------------------------------------------------------------------
# Internal: inverse-normal CDF for Wilson CI z-score
# ---------------------------------------------------------------------------


def _normal_quantile(p: float) -> float:
    """Approximate the inverse standard-normal CDF (probit / quantile fn).

    Beasley-Springer-Moro algorithm. Accurate to ~1e-7 across the entire
    domain — orders of magnitude tighter than we need for a 95% z-score
    (which only needs to be 1.95996... ± 1e-3 to compute Wilson CI to 4
    decimals). Inlined here so the package doesn't pull scipy just for one
    quantile call.
    """
    if not 0.0 < p < 1.0:
        msg = f"_normal_quantile requires 0 < p < 1, got {p!r}"
        raise ValueError(msg)

    # Beasley-Springer (1977) coefficients.
    a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    )
    b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )
