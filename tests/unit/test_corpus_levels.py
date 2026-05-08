"""Unit tests for the four Corpus granularity levels."""

from __future__ import annotations

import pytest
from kaos_content.model.blocks import Heading, Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata, SourceRef

from kaos_ml_core import Corpus, CorpusError

pytestmark = pytest.mark.unit


def _doc_with_sections() -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(source=SourceRef(uri="contract://test/1")),
        body=(
            Heading(depth=1, children=(Text(value="Indemnification"),)),
            Paragraph(children=(Text(value="The Seller indemnifies Buyer."),)),
            Paragraph(children=(Text(value="Cap is 10% of purchase price."),)),
            Heading(depth=1, children=(Text(value="Governing Law"),)),
            Paragraph(children=(Text(value="Governed by Delaware law."),)),
        ),
    )


def _doc_no_headings() -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(source=SourceRef(uri="contract://test/2")),
        body=(
            Paragraph(children=(Text(value="A solo paragraph."),)),
            Paragraph(children=(Text(value="A second one."),)),
        ),
    )


class TestParagraphLevel:
    def test_paragraph_one_unit_per_paragraph(self):
        c = Corpus.from_documents([_doc_with_sections()], level="paragraph")
        assert len(c) == 3  # three paragraphs

    def test_paragraph_carries_section_metadata(self):
        c = Corpus.from_documents([_doc_with_sections()], level="paragraph")
        assert c.unit(0).section_title == "Indemnification"
        assert c.unit(2).section_title == "Governing Law"


class TestSectionLevel:
    def test_section_groups_by_section_ref(self):
        c = Corpus.from_documents([_doc_with_sections()], level="section")
        assert len(c) == 2  # two sections

    def test_section_concatenates_paragraphs(self):
        c = Corpus.from_documents([_doc_with_sections()], level="section")
        # Indemnification section has both indemnification paragraphs joined.
        assert "Seller indemnifies" in c.unit(0).text
        assert "Cap is 10%" in c.unit(0).text
        # And the section_title is preserved.
        assert c.unit(0).section_title == "Indemnification"

    def test_section_emits_one_per_section(self):
        c = Corpus.from_documents([_doc_with_sections()], level="section")
        section_refs = [c.unit(i).section_ref for i in range(len(c))]
        # Two distinct section_refs.
        assert len(section_refs) == 2

    def test_no_headings_emits_ungrouped_units(self):
        # When a doc has no section structure, each paragraph becomes its
        # own ungrouped section unit.
        c = Corpus.from_documents([_doc_no_headings()], level="section")
        assert len(c) == 2  # two ungrouped paragraphs

    def test_multi_doc_section_isolation(self):
        # Each doc's sections stay scoped to that doc.
        c = Corpus.from_documents([_doc_with_sections(), _doc_no_headings()], level="section")
        assert len(c) == 4  # 2 sections + 2 ungrouped


class TestDocumentLevel:
    def test_document_one_row_per_doc(self):
        c = Corpus.from_documents([_doc_with_sections(), _doc_no_headings()], level="document")
        assert len(c) == 2

    def test_document_block_ref_is_root(self):
        c = Corpus.from_documents([_doc_with_sections()], level="document")
        assert c.unit(0).block_ref == "#"


class TestUnknownLevel:
    def test_unknown_level_raises(self):
        with pytest.raises(CorpusError, match=r"Unknown level"):
            Corpus.from_documents([_doc_with_sections()], level="clause")

    def test_error_lists_supported_levels(self):
        with pytest.raises(CorpusError) as exc_info:
            Corpus.from_documents([_doc_with_sections()], level="word")
        msg = str(exc_info.value)
        assert "paragraph" in msg
        assert "sentence" in msg
        assert "section" in msg
        assert "document" in msg


class TestRowReindexing:
    def test_rows_are_dense_zero_indexed(self):
        c = Corpus.from_documents([_doc_with_sections()], level="paragraph")
        # Rows 0..N-1, no gaps.
        for i in range(len(c)):
            assert c.unit(i).row == i
