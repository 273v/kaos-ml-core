"""Benchmarks for Corpus — construction and lookup performance.

Uses pytest-benchmark to measure hot-path operations at various corpus
sizes. Run with:

    pytest tests/benchmarks --benchmark-only -v

Or to compare against a saved baseline:

    pytest tests/benchmarks --benchmark-compare
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(paragraph_texts: list[str], doc_uri: str = "test://bench/1"):
    """Build a synthetic ContentDocument with the given paragraph texts."""
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


def _make_n_paragraph_doc(n: int, doc_uri: str = "test://bench/1"):
    """Build a synthetic doc with n paragraphs of realistic-ish text."""
    texts = [
        f"Paragraph {i}: This is a synthetic paragraph with enough content "
        f"to be realistic for benchmarking purposes. It contains several "
        f"sentences worth of text to exercise the pipeline properly."
        for i in range(n)
    ]
    return _make_doc(texts, doc_uri=doc_uri)


def _build_corpus(n: int):
    """Build a paragraph-level Corpus with n units."""
    from kaos_ml_core import Corpus

    doc = _make_n_paragraph_doc(n)
    return Corpus.from_paragraphs(doc)


# ---------------------------------------------------------------------------
# Construction benchmarks
# ---------------------------------------------------------------------------


def test_bench_construction_100(benchmark):
    """Benchmark: build Corpus from 100-paragraph doc."""
    from kaos_ml_core import Corpus

    doc = _make_n_paragraph_doc(100)
    benchmark(Corpus.from_paragraphs, doc)


def test_bench_construction_1000(benchmark):
    """Benchmark: build Corpus from 1000-paragraph doc."""
    from kaos_ml_core import Corpus

    doc = _make_n_paragraph_doc(1000)
    benchmark(Corpus.from_paragraphs, doc)


def test_bench_construction_10000(benchmark):
    """Benchmark: build Corpus from 10000-paragraph doc."""
    from kaos_ml_core import Corpus

    doc = _make_n_paragraph_doc(10000)
    benchmark(Corpus.from_paragraphs, doc)


# ---------------------------------------------------------------------------
# Lookup benchmarks
# ---------------------------------------------------------------------------


def test_bench_unit_lookup(benchmark):
    """Benchmark: time unit(row) lookup at various corpus sizes."""
    corpus = _build_corpus(1000)
    # Lookup the middle element — avoids cache-line effects at boundaries
    target = len(corpus) // 2

    benchmark(corpus.unit, target)


def test_bench_row_for_lookup(benchmark):
    """Benchmark: time row_for(block_ref) lookup."""
    corpus = _build_corpus(1000)
    target_ref = corpus.unit(len(corpus) // 2).block_ref

    benchmark(corpus.row_for, target_ref)


def test_bench_rows_for_lookup(benchmark):
    """Benchmark: time rows_for(block_ref) for paragraph-level corpus.

    At paragraph level each block_ref maps to a single row; this measures
    the dictionary lookup + list copy path.
    """
    corpus = _build_corpus(1000)
    target_ref = corpus.unit(len(corpus) // 2).block_ref

    benchmark(corpus.rows_for, target_ref)


def test_bench_units_for_doc(benchmark):
    """Benchmark: time units_for_doc(uri) at 1000 units.

    This is a linear scan, so it should scale with corpus size.
    """
    corpus = _build_corpus(1000)

    benchmark(corpus.units_for_doc, "test://bench/1")
