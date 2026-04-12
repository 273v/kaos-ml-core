"""Unit tests for kaos_ml_core.predict — classifier application.

Tests predict_corpus with a trained LogisticRegression and a Corpus
built from synthetic ContentDocuments.
"""

from __future__ import annotations

import numpy as np
import pytest

from kaos_ml_core.corpus import Corpus
from kaos_ml_core.errors import PredictError
from kaos_ml_core.predict import predict_corpus
from kaos_ml_core.train import train_logreg

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(texts: list[str], doc_uri: str = "test://predict"):
    from kaos_content.model.attr import SourceRef
    from kaos_content.model.blocks import Paragraph
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.inlines import Text
    from kaos_content.model.metadata import DocumentMetadata

    paragraphs = tuple(Paragraph(children=(Text(value=t),)) for t in texts if t.strip())
    return ContentDocument(
        metadata=DocumentMetadata(source=SourceRef(uri=doc_uri)),
        body=paragraphs,
    )


def _make_corpus_and_features(n: int = 20):
    """Build a corpus with n paragraphs and a matching feature matrix."""
    doc = _make_doc([f"Paragraph number {i} about topic." for i in range(n)])
    corpus = Corpus.from_paragraphs(doc)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(len(corpus), 8)).astype(np.float32)
    return corpus, X


def _train_clf(X: np.ndarray, n_pos: int = 5):
    """Train a simple binary classifier on the first rows."""
    labels = {}
    for i in range(min(10, X.shape[0])):
        labels[i] = "positive" if i < n_pos else "negative"
    return train_logreg(X, labels)


# ---------------------------------------------------------------------------
# predict_corpus
# ---------------------------------------------------------------------------


class TestPredictCorpus:
    def test_basic(self) -> None:
        corpus, X = _make_corpus_and_features()
        clf = _train_clf(X)
        result = predict_corpus(corpus, X, clf)

        # Returns a TabularDocument
        from kaos_content.model.tabular import TabularDocument

        assert isinstance(result, TabularDocument)
        assert len(result.tables) == 1
        table = result.tables[0]
        assert table.name == "predictions"
        assert len(table.rows) == len(corpus)

    def test_columns(self) -> None:
        corpus, X = _make_corpus_and_features()
        clf = _train_clf(X)
        result = predict_corpus(corpus, X, clf)
        table = result.tables[0]
        col_names = [c.name for c in table.columns]
        expected = [
            "row", "block_ref", "doc_uri", "page",
            "section_ref", "section_title", "predicted_label",
            "score", "above_threshold",
        ]
        assert col_names == expected

    def test_row_indices_match_corpus(self) -> None:
        corpus, X = _make_corpus_and_features()
        clf = _train_clf(X)
        result = predict_corpus(corpus, X, clf)
        table = result.tables[0]
        for i, row in enumerate(table.rows):
            assert row[0] == i  # row index
            assert row[1] == corpus.unit(i).block_ref  # block_ref

    def test_scores_are_probabilities(self) -> None:
        corpus, X = _make_corpus_and_features()
        clf = _train_clf(X)
        result = predict_corpus(corpus, X, clf)
        table = result.tables[0]
        for row in table.rows:
            score = row[7]  # score column
            assert 0.0 <= score <= 1.0

    def test_above_threshold(self) -> None:
        corpus, X = _make_corpus_and_features()
        clf = _train_clf(X)
        result = predict_corpus(corpus, X, clf, threshold=0.5)
        table = result.tables[0]
        for row in table.rows:
            score = row[7]
            above = row[8]
            assert above == (score >= 0.5)

    def test_custom_positive_label(self) -> None:
        corpus, X = _make_corpus_and_features()
        clf = _train_clf(X)
        result = predict_corpus(corpus, X, clf, positive_label="negative")
        # Should succeed without error
        assert len(result.tables[0].rows) == len(corpus)

    def test_rejects_shape_mismatch(self) -> None:
        corpus, X = _make_corpus_and_features()
        clf = _train_clf(X)
        bad_X = X[:5]
        with pytest.raises(PredictError, match="rows"):
            predict_corpus(corpus, bad_X, clf)

    def test_rejects_no_predict_proba(self) -> None:
        corpus, X = _make_corpus_and_features()

        class FakeClf:
            classes_ = ["a", "b"]

        with pytest.raises(PredictError, match="predict_proba"):
            predict_corpus(corpus, X, FakeClf())

    def test_rejects_invalid_positive_label(self) -> None:
        corpus, X = _make_corpus_and_features()
        clf = _train_clf(X)
        with pytest.raises(PredictError, match="positive_label"):
            predict_corpus(corpus, X, clf, positive_label="nonexistent")

    def test_doc_uri_in_output(self) -> None:
        corpus, X = _make_corpus_and_features()
        clf = _train_clf(X)
        result = predict_corpus(corpus, X, clf)
        table = result.tables[0]
        for row in table.rows:
            assert row[2] == "test://predict"  # doc_uri column
