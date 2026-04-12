"""Apply a trained classifier to a Corpus and emit a TabularDocument.

This module closes the AST-grounding round-trip: every prediction is
joined back to its CorpusUnit by row index, carrying the ``block_ref``
that resolves through ``DocumentView`` to the original paragraph in the
source ContentDocument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from kaos_ml_core.corpus import Corpus
from kaos_ml_core.errors import PredictError

if TYPE_CHECKING:
    from kaos_content.model.tabular import TabularDocument


def predict_corpus(
    corpus: Corpus,
    X: np.ndarray,
    clf: Any,
    *,
    threshold: float = 0.5,
    positive_label: str | None = None,
) -> TabularDocument:
    """Apply a fitted classifier to every row in a Corpus.

    Returns a ``TabularDocument`` with one row per ``CorpusUnit``,
    joined by row index, carrying the AST ``block_ref`` for every
    prediction. The invariants in PRD §5 are preserved: every
    prediction round-trips back to a paragraph (or sentence) in the
    source ContentDocument.

    Args:
        corpus: The Corpus that produced X.
        X: Feature matrix, shape ``(len(corpus), D)``, row-aligned with
            ``corpus``.
        clf: Fitted sklearn classifier exposing ``predict_proba`` and
            ``classes_``.
        threshold: Decision threshold on the positive class score.
        positive_label: Which class to call "positive". Defaults to
            ``clf.classes_[1]`` (the second class — the standard sklearn
            binary convention).

    Returns:
        TabularDocument with columns:
            ``row``, ``block_ref``, ``doc_uri``, ``page``,
            ``section_ref``, ``section_title``, ``predicted_label``,
            ``score``, ``above_threshold``

    Raises:
        PredictError: On shape mismatch, missing predict_proba, or
            invalid positive_label.
    """
    if X.shape[0] != len(corpus):
        msg = (
            f"X has {X.shape[0]} rows but corpus has {len(corpus)} units. "
            "Fix: rebuild the feature matrix from the same Corpus."
        )
        raise PredictError(msg)

    if not hasattr(clf, "predict_proba"):
        msg = (
            f"Classifier {type(clf).__name__} does not support predict_proba. "
            "Fix: use LogisticRegression (the v0 default) or wrap your classifier "
            "in CalibratedClassifierCV. "
            "Alternative: in Phase v1.4 LinearSVC + sigmoid calibration is wired "
            "via train_classifier(model='linearsvc')."
        )
        raise PredictError(msg)

    classes = list(clf.classes_)
    if len(classes) < 2:
        msg = (
            f"Classifier has {len(classes)} classes; need at least 2. "
            "Fix: train on at least 2 distinct labels."
        )
        raise PredictError(msg)

    pos = positive_label if positive_label is not None else classes[1]
    if pos not in classes:
        msg = (
            f"positive_label={pos!r} not in clf.classes_={classes}. "
            "Fix: pick one of the trained classes."
        )
        raise PredictError(msg)
    pos_idx = classes.index(pos)

    proba = clf.predict_proba(X)
    scores = proba[:, pos_idx]
    pred_idx = np.argmax(proba, axis=1)
    predicted_labels = np.array(classes)[pred_idx]
    above = scores >= threshold

    from kaos_content.model.tabular import (
        Column,
        ColumnType,
        Table,
        TabularDocument,
    )

    columns = (
        Column(name="row", column_type=ColumnType.INTEGER),
        Column(name="block_ref", column_type=ColumnType.TEXT),
        Column(name="doc_uri", column_type=ColumnType.TEXT),
        Column(name="page", column_type=ColumnType.INTEGER),
        Column(name="section_ref", column_type=ColumnType.TEXT),
        Column(name="section_title", column_type=ColumnType.TEXT),
        Column(name="predicted_label", column_type=ColumnType.TEXT),
        Column(name="score", column_type=ColumnType.FLOAT),
        Column(name="above_threshold", column_type=ColumnType.BOOLEAN),
    )
    rows = tuple(
        (
            u.row,
            u.block_ref,
            u.doc_uri,
            u.page,
            u.section_ref,
            u.section_title,
            str(predicted_labels[i]),
            float(scores[i]),
            bool(above[i]),
        )
        for i, u in enumerate(corpus)
    )
    table = Table(name="predictions", columns=columns, rows=rows)
    return TabularDocument(tables=(table,))


__all__ = ["predict_corpus"]
