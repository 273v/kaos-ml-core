"""Unit tests for kaos_ml_core.features — feature matrix construction.

In v0, embed_corpus requires the [transformers] extra (kaos-nlp-transformers)
which may not be installed. tfidf_corpus is not implemented in v0.
These tests verify the API contracts and error handling.
"""

from __future__ import annotations

import importlib

import pytest

from kaos_ml_core.corpus import Corpus
from kaos_ml_core.errors import FeatureError
from kaos_ml_core.features import tfidf_corpus

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(texts: list[str], doc_uri: str = "test://features"):
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


def _make_corpus(n: int = 10) -> Corpus:
    doc = _make_doc([f"Document text about topic {i}." for i in range(n)])
    return Corpus.from_paragraphs(doc)


# ---------------------------------------------------------------------------
# tfidf_corpus (v0: NotImplementedError)
# ---------------------------------------------------------------------------


class TestTfidfCorpus:
    def test_raises_not_implemented(self) -> None:
        corpus = _make_corpus()
        with pytest.raises(NotImplementedError, match="not implemented in v0"):
            tfidf_corpus(corpus)

    def test_error_message_suggests_alternative(self) -> None:
        corpus = _make_corpus()
        with pytest.raises(NotImplementedError, match="embed_corpus"):
            tfidf_corpus(corpus)


# ---------------------------------------------------------------------------
# embed_corpus (requires [transformers] extra)
# ---------------------------------------------------------------------------


class TestEmbedCorpus:
    def test_import_error_without_transformers(self) -> None:
        """embed_corpus should raise FeatureError when kaos-nlp-transformers
        is not installed."""
        try:
            importlib.import_module("kaos_nlp_transformers")
            pytest.skip("kaos-nlp-transformers is installed; cannot test import error")
        except ImportError:
            pass

        from kaos_ml_core.features import embed_corpus

        corpus = _make_corpus()
        with pytest.raises(FeatureError, match="transformers"):
            embed_corpus(corpus)

    def test_embed_corpus_with_transformers(self) -> None:
        """If kaos-nlp-transformers is installed, embed_corpus should
        produce a valid feature matrix."""
        try:
            importlib.import_module("kaos_nlp_transformers")
        except ImportError:
            pytest.skip("kaos-nlp-transformers not installed")

        import numpy as np

        from kaos_ml_core.features import embed_corpus

        corpus = _make_corpus(5)
        X = embed_corpus(corpus)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(corpus)
        assert X.ndim == 2
        assert X.dtype == np.float32
