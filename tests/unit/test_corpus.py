"""Unit tests for Corpus — the AST-grounding round-trip invariants.

These tests validate the five invariants in PRD §5 against synthetic
ContentDocuments. The integration test in
``tests/integration/test_v0_vertical_slice.py`` validates the same
invariants against real extracted PDFs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(paragraph_texts: list[str], doc_uri: str = "test://doc/1"):
    from kaos_content.model.attr import SourceRef
    from kaos_content.model.blocks import Paragraph
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.inlines import Text
    from kaos_content.model.metadata import DocumentMetadata

    paragraphs = tuple(Paragraph(children=(Text(value=t),)) for t in paragraph_texts if t.strip())
    return ContentDocument(
        metadata=DocumentMetadata(source=SourceRef(uri=doc_uri)),
        body=paragraphs,
    )


def _make_sentence_units(paragraph_texts: list[str], doc_uri: str = "test://doc/1"):
    """Build synthetic SentenceUnit objects for mocking iter_sentence_units.

    Splits each paragraph text on '. ' to produce sentences. Each sentence
    carries the block_ref of its containing paragraph.
    """
    from kaos_content.units import SentenceUnit

    units = []
    row = 0
    for para_idx, para_text in enumerate(paragraph_texts):
        if not para_text.strip():
            continue
        block_ref = f"#/body/{para_idx}"
        # Split on '. ' to simulate sentence segmentation
        start = 0
        while start < len(para_text):
            end = para_text.find(". ", start)
            if end == -1:
                sent_text = para_text[start:]
                char_end = len(para_text)
            else:
                sent_text = para_text[start : end + 1]
                char_end = end + 1
            if sent_text.strip():
                units.append(
                    SentenceUnit(
                        row=row,
                        text=sent_text,
                        block_ref=block_ref,
                        page=None,
                        section_ref=None,
                        section_title=None,
                        char_start=start,
                        char_end=char_end,
                    )
                )
                row += 1
            if end == -1:
                break
            start = end + 2
    return units


# ---------------------------------------------------------------------------
# Basic construction (existing tests)
# ---------------------------------------------------------------------------


def test_from_paragraphs_basic():
    from kaos_ml_core import Corpus

    doc = _make_doc(["First paragraph.", "Second paragraph.", "Third paragraph."])
    corpus = Corpus.from_paragraphs(doc)

    assert len(corpus) == 3
    for r in range(len(corpus)):
        unit = corpus.unit(r)
        assert unit.row == r
        assert unit.doc_uri == "test://doc/1"
        assert unit.text


def test_skips_empty_paragraphs():
    from kaos_ml_core import Corpus

    doc = _make_doc(["Real text.", "  ", "More real text.", ""])
    corpus = Corpus.from_paragraphs(doc)
    # Empty / whitespace-only paragraphs are skipped (mirrors search.py)
    assert len(corpus) == 2


def test_invariant_row_equals_index():
    """Invariant 1: corpus.unit(r).row == r"""
    from kaos_ml_core import Corpus

    doc = _make_doc([f"Paragraph {i}." for i in range(10)])
    corpus = Corpus.from_paragraphs(doc)
    for r in range(len(corpus)):
        assert corpus.unit(r).row == r


def test_invariant_block_ref_round_trip():
    """Invariants 2 & 3: row <-> block_ref bidirectional."""
    from kaos_ml_core import Corpus

    doc = _make_doc([f"Paragraph {i}." for i in range(10)])
    corpus = Corpus.from_paragraphs(doc)

    for r in range(len(corpus)):
        block_ref = corpus.block_ref_for(r)
        # Round-trip: block_ref -> row -> block_ref
        assert corpus.row_for(block_ref) == r
        assert corpus.unit(r).block_ref == block_ref


def test_block_refs_are_unique_at_paragraph_level():
    """At paragraph granularity each row should have a unique block_ref."""
    from kaos_ml_core import Corpus

    doc = _make_doc([f"Paragraph {i}." for i in range(10)])
    corpus = Corpus.from_paragraphs(doc)
    refs = {u.block_ref for u in corpus}
    assert len(refs) == len(corpus)


def test_rows_for_returns_all_matching():
    from kaos_ml_core import Corpus

    doc = _make_doc(["Para A.", "Para B."])
    corpus = Corpus.from_paragraphs(doc)
    for r in range(len(corpus)):
        rows = corpus.rows_for(corpus.unit(r).block_ref)
        assert rows == [r]


def test_multi_doc_corpus_dense_row_indices():
    from kaos_ml_core import Corpus

    doc1 = _make_doc(["A1.", "A2."], doc_uri="test://A")
    doc2 = _make_doc(["B1.", "B2.", "B3."], doc_uri="test://B")
    corpus = Corpus.from_documents([doc1, doc2], level="paragraph")

    assert len(corpus) == 5
    for r in range(5):
        assert corpus.unit(r).row == r

    # Per-doc filtering
    a_rows = corpus.units_for_doc("test://A")
    b_rows = corpus.units_for_doc("test://B")
    assert a_rows == [0, 1]
    assert b_rows == [2, 3, 4]


def test_explicit_doc_uri_override():
    from kaos_content.model.blocks import Paragraph
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.inlines import Text
    from kaos_content.model.metadata import DocumentMetadata

    from kaos_ml_core import Corpus

    # Document with NO source set
    doc = ContentDocument(
        metadata=DocumentMetadata(),
        body=(Paragraph(children=(Text(value="Hello."),)),),
    )
    corpus = Corpus.from_documents(
        [doc],
        level="paragraph",
        doc_uris=["override://only"],
    )
    assert corpus.unit(0).doc_uri == "override://only"


def test_missing_doc_uri_raises():
    from kaos_content.model.blocks import Paragraph
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.inlines import Text
    from kaos_content.model.metadata import DocumentMetadata

    from kaos_ml_core import Corpus, CorpusError

    doc = ContentDocument(
        metadata=DocumentMetadata(),
        body=(Paragraph(children=(Text(value="Hello."),)),),
    )
    with pytest.raises(CorpusError, match=r"metadata\.source\.uri"):
        Corpus.from_documents([doc], level="paragraph")


def test_empty_corpus_raises():
    from kaos_ml_core import Corpus, CorpusError

    doc = _make_doc(["", "   ", "\n"])
    with pytest.raises(CorpusError, match="empty"):
        Corpus.from_paragraphs(doc)


def test_unknown_level_raises():
    from kaos_ml_core import Corpus, CorpusError

    doc = _make_doc(["Hello."])
    with pytest.raises(CorpusError, match="Unknown level"):
        Corpus.from_documents([doc], level="bogus")


def test_to_tabular_round_trip():
    from kaos_ml_core import Corpus

    doc = _make_doc([f"Para {i}." for i in range(5)])
    corpus = Corpus.from_paragraphs(doc)
    tab = corpus.to_tabular()

    assert len(tab.tables) == 1
    table = tab.tables[0]
    assert table.name == "corpus"
    assert len(table.rows) == 5
    # Column 0 is `row`, column 1 is `block_ref`
    for i, row in enumerate(table.rows):
        assert row[0] == i
        assert row[1] == corpus.block_ref_for(i)


# ---------------------------------------------------------------------------
# Sentence-level corpus tests
# ---------------------------------------------------------------------------


class TestSentenceLevelCorpus:
    """Tests for sentence-level Corpus construction and invariants."""

    def test_from_sentences_basic(self):
        """Build a sentence-level corpus, verify units have char_start/char_end."""
        from kaos_ml_core import Corpus

        para_texts = ["First sentence. Second sentence.", "Third sentence."]
        sentence_units = _make_sentence_units(para_texts)

        with patch("kaos_content.units.iter_sentence_units", return_value=sentence_units):
            doc = _make_doc(para_texts)
            corpus = Corpus.from_sentences(doc)

        # Should have 3 sentences
        assert len(corpus) == 3
        for r in range(len(corpus)):
            unit = corpus.unit(r)
            assert unit.row == r
            assert unit.text
            assert unit.doc_uri == "test://doc/1"

    def test_sentence_rows_for_returns_multiple(self):
        """Multiple sentences share one paragraph block_ref; verify rows_for returns all."""
        from kaos_ml_core import Corpus

        para_texts = ["First sentence. Second sentence. Third sentence."]
        sentence_units = _make_sentence_units(para_texts)

        with patch("kaos_content.units.iter_sentence_units", return_value=sentence_units):
            doc = _make_doc(para_texts)
            corpus = Corpus.from_sentences(doc)

        # All sentences share the same paragraph block_ref (#/body/0)
        block_ref = corpus.unit(0).block_ref
        rows = corpus.rows_for(block_ref)
        assert len(rows) == 3
        assert rows == [0, 1, 2]

        # row_for returns the FIRST row
        assert corpus.row_for(block_ref) == 0

    def test_sentence_level_dense_indices(self):
        """Verify row indices are still dense 0..N for sentence-level."""
        from kaos_ml_core import Corpus

        para_texts = ["A. B.", "C. D. E."]
        sentence_units = _make_sentence_units(para_texts)

        with patch("kaos_content.units.iter_sentence_units", return_value=sentence_units):
            doc = _make_doc(para_texts)
            corpus = Corpus.from_sentences(doc)

        for i in range(len(corpus)):
            assert corpus.unit(i).row == i


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for Corpus construction and access."""

    def test_single_paragraph_document(self):
        """One-paragraph doc, verify corpus has 1 unit."""
        from kaos_ml_core import Corpus

        doc = _make_doc(["The only paragraph."])
        corpus = Corpus.from_paragraphs(doc)
        assert len(corpus) == 1
        assert corpus.unit(0).text == "The only paragraph."

    def test_document_with_many_paragraphs(self):
        """100-paragraph doc, verify all indexed."""
        from kaos_ml_core import Corpus

        texts = [f"Paragraph number {i}." for i in range(100)]
        doc = _make_doc(texts)
        corpus = Corpus.from_paragraphs(doc)
        assert len(corpus) == 100
        for r in range(100):
            assert corpus.unit(r).row == r
            assert f"Paragraph number {r}." in corpus.unit(r).text

    def test_empty_document_in_multi_doc(self):
        """One doc has paragraphs, other doesn't; verify corpus only has units from the first."""
        from kaos_ml_core import Corpus

        doc1 = _make_doc(["Hello.", "World."], doc_uri="test://A")
        # Build doc2 with an empty body so it contributes 0 units.
        from kaos_content.model.attr import SourceRef
        from kaos_content.model.document import ContentDocument
        from kaos_content.model.metadata import DocumentMetadata

        doc2_empty = ContentDocument(
            metadata=DocumentMetadata(source=SourceRef(uri="test://B")),
            body=(),
        )

        corpus = Corpus.from_documents([doc1, doc2_empty], level="paragraph")
        # Only doc1's 2 paragraphs should be in the corpus
        assert len(corpus) == 2
        for u in corpus:
            assert u.doc_uri == "test://A"

    def test_negative_row_index_raises(self):
        """unit(-1) should raise IndexError."""
        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello."])
        corpus = Corpus.from_paragraphs(doc)
        with pytest.raises(IndexError, match="out of range"):
            corpus.unit(-1)

    def test_out_of_range_row_index_raises(self):
        """unit(len+1) should raise IndexError."""
        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello.", "World."])
        corpus = Corpus.from_paragraphs(doc)
        with pytest.raises(IndexError, match="out of range"):
            corpus.unit(len(corpus))
        with pytest.raises(IndexError, match="out of range"):
            corpus.unit(len(corpus) + 1)


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------


class TestCaching:
    """Tests for embed() and retriever() caching behavior."""

    def test_embed_caches_by_model(self):
        """Call embed twice with same model, verify same object returned."""
        import numpy as np

        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello.", "World."])
        corpus = Corpus.from_paragraphs(doc)

        fake_embeddings = np.random.randn(len(corpus), 16).astype(np.float32)

        with patch("kaos_ml_core.features.embed_corpus", return_value=fake_embeddings):
            result1 = corpus.embed(model="test-model")
            result2 = corpus.embed(model="test-model")

        # Same object — cached, not recomputed
        assert result1 is result2

    def test_embed_different_model_produces_different_cache(self):
        """Call embed with model='A' then model='B', verify different arrays."""
        import numpy as np

        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello.", "World."])
        corpus = Corpus.from_paragraphs(doc)

        fake_a = np.random.randn(len(corpus), 16).astype(np.float32)
        fake_b = np.random.randn(len(corpus), 32).astype(np.float32)

        call_count = 0

        def mock_embed(corpus_arg, *, model=None, batch_size=32):
            nonlocal call_count
            call_count += 1
            if model == "model-A":
                return fake_a
            return fake_b

        with patch("kaos_ml_core.features.embed_corpus", side_effect=mock_embed):
            result_a = corpus.embed(model="model-A")
            result_b = corpus.embed(model="model-B")

        assert result_a is not result_b
        assert result_a is fake_a
        assert result_b is fake_b
        assert call_count == 2

    def test_retriever_caches_by_method_and_group_by(self):
        """Call retriever('bm25') twice, verify same object returned."""
        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello.", "World."])
        corpus = Corpus.from_paragraphs(doc)

        fake_retriever = MagicMock()

        with patch(
            "kaos_nlp_core.retrieval.bm25.BM25Retriever.from_corpus",
            return_value=fake_retriever,
        ):
            r1 = corpus.retriever("bm25")
            r2 = corpus.retriever("bm25")

        assert r1 is r2
        assert r1 is fake_retriever

    def test_retriever_different_kwargs_produces_different_cache(self):
        """Verify different kwargs produce different cached retrievers."""
        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello.", "World."])
        corpus = Corpus.from_paragraphs(doc)

        fake_ret_a = MagicMock(name="retriever-a")
        fake_ret_b = MagicMock(name="retriever-b")

        call_count = 0

        def mock_from_corpus(corpus_arg, *, group_by=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("k") == 10:
                return fake_ret_a
            return fake_ret_b

        with patch(
            "kaos_nlp_core.retrieval.bm25.BM25Retriever.from_corpus",
            side_effect=mock_from_corpus,
        ):
            ra = corpus.retriever("bm25", k=10)
            rb = corpus.retriever("bm25", k=20)

        assert ra is not rb
        assert ra is fake_ret_a
        assert rb is fake_ret_b
        assert call_count == 2

    def test_corpus_metadata_returns_copy(self):
        """Modify the returned dict, verify internal state unchanged."""
        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello.", "World."])
        corpus = Corpus.from_paragraphs(doc)

        meta1 = corpus.corpus_metadata
        meta1["injected_key"] = "should not persist"

        meta2 = corpus.corpus_metadata
        assert "injected_key" not in meta2


# ---------------------------------------------------------------------------
# corpus_metadata tests
# ---------------------------------------------------------------------------


class TestCorpusMetadata:
    """Tests for the corpus_metadata property."""

    def test_corpus_metadata_has_all_fields(self):
        """Verify level, doc_count, unit_count, doc_uris are all present."""
        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello.", "World."])
        corpus = Corpus.from_paragraphs(doc)

        meta = corpus.corpus_metadata
        assert "level" in meta
        assert "doc_count" in meta
        assert "unit_count" in meta
        assert "doc_uris" in meta

        assert meta["level"] == "paragraph"
        assert meta["doc_count"] == 1
        assert meta["unit_count"] == 2

    def test_corpus_metadata_doc_uris_all_unique(self):
        """With 3 multi-paragraph docs, verify all 3 URIs in doc_uris."""
        from kaos_ml_core import Corpus

        doc1 = _make_doc(["A1.", "A2."], doc_uri="test://A")
        doc2 = _make_doc(["B1.", "B2."], doc_uri="test://B")
        doc3 = _make_doc(["C1.", "C2."], doc_uri="test://C")

        corpus = Corpus.from_documents([doc1, doc2, doc3], level="paragraph")
        meta = corpus.corpus_metadata

        assert set(meta["doc_uris"]) == {"test://A", "test://B", "test://C"}
        assert len(meta["doc_uris"]) == 3

    def test_corpus_metadata_doc_count_correct(self):
        """Verify doc_count matches number of input documents."""
        from kaos_ml_core import Corpus

        docs = [_make_doc([f"Doc {i}."], doc_uri=f"test://doc/{i}") for i in range(5)]
        corpus = Corpus.from_documents(docs, level="paragraph")

        meta = corpus.corpus_metadata
        assert meta["doc_count"] == 5


# ---------------------------------------------------------------------------
# to_tabular tests
# ---------------------------------------------------------------------------


class TestToTabular:
    """Tests for the to_tabular() method."""

    def test_to_tabular_columns_correct(self):
        """Verify column names and types."""
        from kaos_content.model.tabular import ColumnType

        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello.", "World."])
        corpus = Corpus.from_paragraphs(doc)
        tab = corpus.to_tabular()
        table = tab.tables[0]

        expected_columns = [
            "row",
            "block_ref",
            "doc_uri",
            "page",
            "section_ref",
            "section_title",
            "text",
        ]
        actual_columns = [c.name for c in table.columns]
        assert actual_columns == expected_columns

        # Verify types for key columns
        col_map = {c.name: c.column_type for c in table.columns}
        assert col_map["row"] == ColumnType.INTEGER
        assert col_map["block_ref"] == ColumnType.TEXT
        assert col_map["doc_uri"] == ColumnType.TEXT
        assert col_map["page"] == ColumnType.INTEGER
        assert col_map["text"] == ColumnType.TEXT

    def test_to_tabular_all_fields_populated(self):
        """Verify every field in every row is populated (no Nones for required fields)."""
        from kaos_ml_core import Corpus

        doc = _make_doc(["Hello world.", "Another paragraph."])
        corpus = Corpus.from_paragraphs(doc)
        tab = corpus.to_tabular()
        table = tab.tables[0]

        for i, row in enumerate(table.rows):
            # row (index 0) — always an int
            assert isinstance(row[0], int)
            assert row[0] == i
            # block_ref (index 1) — always a non-empty string
            assert isinstance(row[1], str)
            assert row[1]
            # doc_uri (index 2) — always a non-empty string
            assert isinstance(row[2], str)
            assert row[2]
            # page (index 3) — may be None (no provenance)
            # section_ref (index 4) — may be None
            # section_title (index 5) — may be None
            # text (index 6) — always a non-empty string
            assert isinstance(row[6], str)
            assert row[6]


# ---------------------------------------------------------------------------
# Performance / scale tests
# ---------------------------------------------------------------------------


class TestPerformanceScale:
    """Tests that verify Corpus works correctly at scale."""

    def test_corpus_construction_1000_paragraphs(self):
        """Build a corpus with 1000 paragraphs, verify it works."""
        from kaos_ml_core import Corpus

        texts = [f"Paragraph number {i} with some content." for i in range(1000)]
        doc = _make_doc(texts)
        corpus = Corpus.from_paragraphs(doc)

        assert len(corpus) == 1000
        # Spot-check first and last
        assert corpus.unit(0).row == 0
        assert corpus.unit(999).row == 999

    def test_corpus_lookup_performance(self):
        """Verify unit(), row_for(), block_ref_for() work at 1000 units."""
        from kaos_ml_core import Corpus

        texts = [f"Paragraph number {i}." for i in range(1000)]
        doc = _make_doc(texts)
        corpus = Corpus.from_paragraphs(doc)

        # Test all three lookup operations at various positions
        for r in [0, 1, 100, 499, 500, 999]:
            unit = corpus.unit(r)
            assert unit.row == r
            assert corpus.block_ref_for(r) == unit.block_ref
            assert corpus.row_for(unit.block_ref) == r

        # rows_for should return single-element lists at paragraph level
        for r in [0, 500, 999]:
            rows = corpus.rows_for(corpus.unit(r).block_ref)
            assert rows == [r]

        # units_for_doc should return all rows for the single doc
        all_rows = corpus.units_for_doc("test://doc/1")
        assert len(all_rows) == 1000
        assert all_rows == list(range(1000))
