"""Classifier training for a labeled subset of a Corpus.

v0: one algorithm only — LogisticRegression(solver="liblinear",
class_weight="balanced"). v1.4 adds LinearSVC + Calibration and
ComplementNB. Random Forest is deliberately excluded with a
NotImplementedError — see ``docs/internal/prd/kaos-ml-core.md`` §14.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kaos_ml_core.errors import TrainError

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression


def train_logreg(
    X: np.ndarray,
    labels: dict[int, str],
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 0,
) -> LogisticRegression:
    """Fit LogisticRegression on the labeled subset of rows.

    Args:
        X: Feature matrix for the entire Corpus, shape ``(N, D)``.
        labels: ``dict[row_index -> label_string]`` for the labeled
            subset. Typically the output of ``label_seeds_with_llm()``.
        C: Inverse regularization strength.
        max_iter: Maximum solver iterations.
        random_state: Random seed.

    Returns:
        A fitted ``LogisticRegression`` with ``classes_`` populated
        from the unique label strings encountered.

    Notes:
        - ``solver="liblinear"`` is the right choice for sparse high-dim
          text and small label sets. **Never use ``lbfgs`` here** — it's
          slower and worse on this regime. See PRD §14.
        - ``class_weight="balanced"`` is mandatory for the typical TAR
          setting where one class is rare.
    """
    if not labels:
        msg = (
            "train_logreg requires at least one labeled row. "
            "Fix: produce labels via label_seeds_with_llm() or supply them directly."
        )
        raise TrainError(msg)

    rows = sorted(labels.keys())
    classes = sorted({labels[r] for r in rows})
    if len(classes) < 2:
        msg = (
            f"train_logreg requires at least 2 distinct classes; got {len(classes)}: {classes}. "
            "Fix: increase per_cluster in kmedoid_seeds(), or pick a more diverse seed set, "
            "or refine the labeling instructions to encourage class balance."
        )
        raise TrainError(msg)

    X_train = X[rows]
    y_train = np.array([labels[r] for r in rows])

    from sklearn.linear_model import LogisticRegression

    # sklearn 1.8 deprecated `penalty=` in favor of `l1_ratio=`. l1_ratio=0
    # corresponds to pure L2 regularization (the v0 default).
    clf = LogisticRegression(
        solver="liblinear",
        l1_ratio=0.0,
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


def train_classifier(
    X: np.ndarray,
    labels: dict[int, str],
    *,
    model: str = "logreg",
    **kwargs,
) -> LogisticRegression:
    """Dispatch to one of the supported training functions.

    v0 supports ``model="logreg"`` only. ``"linearsvc"`` and
    ``"complement_nb"`` land in Phase v1.4. ``"rf"`` (Random Forest)
    raises NotImplementedError — see PRD §14 rule 7.
    """
    if model == "logreg":
        return train_logreg(X, labels, **kwargs)
    if model == "rf":
        msg = (
            "Random Forest on TF-IDF / dense text features is not supported. "
            "Reason: axis-aligned splits on high-dimensional text feature spaces "
            "produce poor classifiers; the TAR-research-backed default is LogReg. "
            "Fix: use train_classifier(model='logreg') (the v0 default). "
            "Alternative: in Phase v1.4, train_classifier(model='linearsvc') will "
            "be available with sigmoid calibration."
        )
        raise NotImplementedError(msg)
    msg = (
        f"Unknown model={model!r}. v0 supports model='logreg' only. "
        "Fix: pass model='logreg', or wait for Phase v1.4 which adds "
        "'linearsvc' and 'complement_nb'."
    )
    raise TrainError(msg)


__all__ = ["train_classifier", "train_logreg"]
