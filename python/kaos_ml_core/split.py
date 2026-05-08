"""Stratified train / test / control splits.

Builds on top of ``sklearn.model_selection.StratifiedShuffleSplit`` to
produce three disjoint subsets:

- **train** — used to fit the classifier
- **test** — used to evaluate (precision/recall/F1)
- **control** — held out for threshold tuning (CLAUDE.md hard rule #5
  forbids tuning on training data; control is the canonical place)

Returns **index arrays** (``np.ndarray[int]``), not data slices, so the
caller's existing data structures (``Corpus``, feature matrix ``X``,
label array ``y``) stay untouched and the same indices can be applied
across all of them. This is the same shape sklearn uses for its built-in
splitters.

Downstream use cases:

- **TAR**: train-test-control is the standard regulatory shape — train
  on labeled examples, evaluate on test, set the operating threshold on
  control to defend a recall claim.
- **Due diligence**: train on a partner-reviewed seed, hold out test +
  control to demonstrate quality, then apply to the full data room.
- **Contract analytics**: control set used to pick the operating point
  for a "binding arbitration present" classifier across thousands of
  contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

__all__ = ["SplitResult", "stratified_split"]


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Index arrays for a stratified train/test/control split.

    All three arrays are ``np.ndarray`` of dtype int64 and are pairwise
    disjoint. ``control_idx`` is empty (length 0) when ``control_frac=0.0``.
    The union of all three is exactly ``range(n)`` (no duplicates, no gaps).
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    control_idx: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)

    @property
    def n_control(self) -> int:
        return int(self.control_idx.size)


def stratified_split(
    labels: np.ndarray,
    *,
    test_frac: float = 0.2,
    control_frac: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    """Stratified train / test / control split that preserves class ratios.

    The split is constructed in two steps to guarantee disjointness while
    keeping all three slices stratified:

    1. Split off ``control_frac`` from the full set (or skip if 0).
    2. Within the remaining ``1 - control_frac``, split off
       ``test_frac / (1 - control_frac)`` as test; the rest is train.

    Both steps stratify on the original labels.

    Args:
        labels: Class labels, shape ``(n,)``. Any dtype that's stable under
            equality comparison (str, int, np.str_).
        test_frac: Fraction of the full set assigned to test. (0, 1).
        control_frac: Fraction held out for threshold tuning. [0, 1).
            Together with ``test_frac`` must leave at least one row for
            train.
        seed: Random seed (passed through to sklearn). Deterministic given
            the seed + label vector.

    Returns:
        ``SplitResult`` with three disjoint, stratified index arrays.

    Raises:
        ValueError: If fractions are out of range, or if the smallest
            class has fewer rows than the number of splits would require.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        msg = f"labels must be 1-D, got ndim={labels.ndim}"
        raise ValueError(msg)
    n = labels.size
    if n < 2:
        msg = f"stratified_split requires at least 2 rows, got {n}"
        raise ValueError(msg)

    if not 0.0 < test_frac < 1.0:
        msg = f"test_frac must be in (0, 1), got {test_frac!r}"
        raise ValueError(msg)
    if not 0.0 <= control_frac < 1.0:
        msg = f"control_frac must be in [0, 1), got {control_frac!r}"
        raise ValueError(msg)
    if test_frac + control_frac >= 1.0:
        msg = (
            f"test_frac ({test_frac}) + control_frac ({control_frac}) must "
            "leave at least some training data; reduce one or both."
        )
        raise ValueError(msg)

    # Smallest-class sanity check: stratified split needs ≥1 row of every
    # class in every output slice. Surface this with a fix-the-data hint
    # rather than letting sklearn emit an opaque "least populated class"
    # warning.
    _classes, class_counts = np.unique(labels, return_counts=True)
    smallest = int(class_counts.min())
    n_required = 2 + (1 if control_frac > 0.0 else 0)
    if smallest < n_required:
        msg = (
            f"smallest class has only {smallest} samples; need at least "
            f"{n_required} (one per train/test"
            f"{'/control' if control_frac > 0.0 else ''} split). "
            "Fix: collect more labels for the rare class, or merge it with "
            "an adjacent class for a binary-by-importance reformulation."
        )
        raise ValueError(msg)

    rng = np.random.RandomState(seed)
    full_idx = np.arange(n, dtype=np.int64)

    if control_frac > 0.0:
        # Step 1: peel off control.
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=control_frac, random_state=rng)
        non_control_idx, control_idx = next(sss1.split(full_idx, labels))
        non_control_idx = full_idx[non_control_idx]
        control_idx = full_idx[control_idx]
        non_control_labels = labels[non_control_idx]
    else:
        non_control_idx = full_idx
        control_idx = np.array([], dtype=np.int64)
        non_control_labels = labels

    # Step 2: split test out of the non-control portion.
    test_frac_within = test_frac / (1.0 - control_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac_within, random_state=rng)
    train_local, test_local = next(sss2.split(non_control_idx, non_control_labels))
    train_idx = non_control_idx[train_local]
    test_idx = non_control_idx[test_local]

    # Stable-sort each output for deterministic downstream behavior.
    train_idx.sort()
    test_idx.sort()
    control_idx.sort()

    return SplitResult(
        train_idx=train_idx,
        test_idx=test_idx,
        control_idx=control_idx,
    )
