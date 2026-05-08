"""Operating-threshold tuning on a held-out control set.

CLAUDE.md hard rule #5: *"Never tune the operating threshold on training
data. Use a held-out control set."* This module enforces that contract:
``tune_threshold`` accepts (y_true, y_proba) for the **control** subset
and returns the threshold that hits a target recall (or precision)
target on it. Pairing with ``stratified_split(..., control_frac=...)``
keeps the discipline at the API level: agentic flows pass
``split.control_idx`` and the math is correct by construction.

Builds on ``sklearn.metrics.precision_recall_curve`` for the threshold
sweep — sklearn returns precision and recall at every distinct probability
the classifier emitted, sorted by threshold. We pick the threshold at the
operating point that satisfies the target.

Downstream use cases:

- **TAR**: regulator says "demonstrate recall ≥ 0.85"; we pick the
  largest threshold that still hits 0.85 on control. This maximizes
  precision (smallest reviewer load) at the recall floor.
- **Due diligence**: partner asks "give me at least 95% recall on
  governing-law clauses across the data room." Same shape.
- **Contract analytics**: partner sets a precision floor instead
  ("don't bother me with junk; precision ≥ 0.90"); we pick the smallest
  threshold that holds the precision target.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve

__all__ = ["ThresholdResult", "tune_threshold"]


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Output of :func:`tune_threshold`.

    Carries the chosen operating threshold AND the achieved
    precision/recall at that point — so callers can immediately log /
    serialize / display the trade-off without recomputing.
    """

    threshold: float
    """Operating threshold in (0, 1). Predictions ``y_proba >= threshold``
    are positive."""

    achieved_recall: float
    """Recall on the control set at this threshold."""

    achieved_precision: float
    """Precision on the control set at this threshold."""

    target_recall: float | None
    """The recall target the user requested (None when target_precision was
    used instead)."""

    target_precision: float | None
    """The precision target the user requested (None when target_recall was
    used)."""

    n_control: int
    """Size of the control set the threshold was tuned on."""


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    target_recall: float | None = None,
    target_precision: float | None = None,
    pos_label: int | str = 1,
) -> ThresholdResult:
    """Find the threshold that achieves a target recall (or precision) on
    a held-out control set.

    **Exactly one** of ``target_recall`` and ``target_precision`` must be
    supplied. If both are None, raises ``ValueError`` (refuses to guess).

    The control set is the subset of (y_true, y_proba) corresponding to
    rows the classifier was NOT trained on. Pairing with
    ``stratified_split(..., control_frac=...)`` makes this trivial:

    .. code-block:: python

        from kaos_ml_core.split import stratified_split
        split = stratified_split(labels, test_frac=0.2, control_frac=0.1)
        clf.fit(X[split.train_idx], labels[split.train_idx])
        proba = clf.predict_proba(X[split.control_idx])[:, 1]
        result = tune_threshold(
            labels[split.control_idx], proba, target_recall=0.85
        )

    Args:
        y_true: Control-set ground-truth labels, shape (n,).
        y_proba: Positive-class probabilities for the control set, shape
            (n,). Must be in [0, 1].
        target_recall: Minimum recall to achieve. Picks the LARGEST
            threshold that still satisfies. Mutually exclusive with
            target_precision; one must be supplied. Typical TAR target:
            0.80-0.95.
        target_precision: Minimum precision to achieve. Picks the SMALLEST
            threshold that still satisfies. Mutually exclusive with
            target_recall.
        pos_label: Positive class label (matches sklearn convention).

    Returns:
        :class:`ThresholdResult`.

    Raises:
        ValueError: On invalid input (length mismatch, both targets set,
            no probability satisfies the target on the control set, etc.).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    if y_true.shape != y_proba.shape:
        msg = (
            f"y_true shape {y_true.shape} must equal y_proba shape "
            f"{y_proba.shape}. Fix: pass the control-set arrays directly; "
            "no slicing in this call."
        )
        raise ValueError(msg)
    if y_true.size == 0:
        msg = "tune_threshold requires a non-empty control set"
        raise ValueError(msg)
    if y_proba.min() < 0.0 or y_proba.max() > 1.0:
        msg = (
            f"y_proba values must be in [0, 1]; got [{y_proba.min():.4g}, "
            f"{y_proba.max():.4g}]. Fix: pass classifier.predict_proba(X)[:, 1] "
            "(the positive-class column), not raw decision-function scores."
        )
        raise ValueError(msg)

    # Both / neither target → refuse.
    if target_recall is None and target_precision is None:
        msg = (
            "tune_threshold requires exactly one of target_recall or "
            "target_precision. Both None means there's no objective to "
            "optimize against."
        )
        raise ValueError(msg)
    if target_recall is not None and target_precision is not None:
        msg = (
            "tune_threshold accepts target_recall OR target_precision, not "
            "both. Pick the one your use case constrains: TAR/ediscovery "
            "typically constrains recall (regulators want all responsive "
            "docs); manual review queues typically constrain precision."
        )
        raise ValueError(msg)
    if target_recall is not None and not 0.0 < target_recall <= 1.0:
        msg = f"target_recall must be in (0, 1], got {target_recall!r}"
        raise ValueError(msg)
    if target_precision is not None and not 0.0 < target_precision <= 1.0:
        msg = f"target_precision must be in (0, 1], got {target_precision!r}"
        raise ValueError(msg)

    # Defensive heuristic for "did you accidentally pass training-set
    # predict() output instead of predict_proba()?" — those are 0/1 binary
    # values which makes threshold tuning a no-op AND silently violates
    # CLAUDE.md hard rule #5 if it's the training set.
    unique_proba = np.unique(y_proba)
    if unique_proba.size <= 2 and set(unique_proba.astype(float)).issubset({0.0, 1.0}):
        warnings.warn(
            "y_proba contains only {0.0, 1.0} values; threshold tuning is a "
            "no-op on hard predictions. Fix: pass clf.predict_proba(X)[:, 1] "
            "(positive-class probabilities), not clf.predict(X) (hard labels). "
            "Hard rule #5: never tune on the training set.",
            RuntimeWarning,
            stacklevel=2,
        )

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba, pos_label=pos_label)
    # precision_recall_curve appends a (precision=1.0, recall=0.0) point to
    # the curve and returns thresholds with one fewer entry — cap them so
    # the three arrays line up for our sweep.
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if thresholds.size == 0:
        msg = (
            "precision_recall_curve returned no thresholds — control set "
            "has only one class. Fix: ensure both classes are present in "
            "the control set (use stratified_split with control_frac > 0)."
        )
        raise ValueError(msg)

    if target_recall is not None:
        # Largest threshold that still hits the recall target.
        # Recall is a non-increasing function of threshold, so we want the
        # largest threshold whose recall is >= target.
        eligible = recalls >= target_recall
        if not eligible.any():
            achieved = float(recalls.max())
            msg = (
                f"No threshold achieves target_recall={target_recall} on the "
                f"control set; max achievable recall is {achieved:.4f}. "
                "Fix: lower the target, expand the labeled training data, "
                "or improve features. Alternative: relabel the control set "
                "if you suspect labeling noise."
            )
            raise ValueError(msg)
        best_idx = int(np.where(eligible)[0].max())
    else:
        # target_precision: smallest threshold whose precision >= target.
        eligible = precisions >= target_precision
        if not eligible.any():
            achieved = float(precisions.max())
            msg = (
                f"No threshold achieves target_precision={target_precision} "
                f"on the control set; max achievable precision is "
                f"{achieved:.4f}. Fix: lower the target or improve features."
            )
            raise ValueError(msg)
        best_idx = int(np.where(eligible)[0].min())

    return ThresholdResult(
        threshold=float(thresholds[best_idx]),
        achieved_recall=float(recalls[best_idx]),
        achieved_precision=float(precisions[best_idx]),
        target_recall=target_recall,
        target_precision=target_precision,
        n_control=int(y_true.size),
    )
