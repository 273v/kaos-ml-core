"""Unit tests for Corpus — the AST-grounding round-trip invariants.

These tests validate the five invariants in PRD §5 against synthetic
ContentDocuments. The integration test in
``tests/integration/test_v0_vertical_slice.py`` validates the same
invariants against real extracted PDFs.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


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
    """Invariants 2 & 3: row ↔ block_ref bidirectional."""
    from kaos_ml_core import Corpus

    doc = _make_doc([f"Paragraph {i}." for i in range(10)])
    corpus = Corpus.from_paragraphs(doc)

    for r in range(len(corpus)):
        block_ref = corpus.block_ref_for(r)
        # Round-trip: block_ref → row → block_ref
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
