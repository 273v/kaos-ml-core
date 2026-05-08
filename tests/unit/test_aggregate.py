"""Unit tests for kaos_ml_core.aggregate — cross-granularity rollup."""

from __future__ import annotations

import pytest
from kaos_content.model.tabular import Column, ColumnType, Table, TabularDocument

from kaos_ml_core import aggregate_predictions

pytestmark = pytest.mark.unit


def _build_predictions(rows: list[tuple]) -> TabularDocument:
    """Construct a predictions TabularDocument from a list of row tuples
    matching the (row, block_ref, doc_uri, page, section_ref,
    section_title, predicted_label, score, above_threshold) shape that
    predict_corpus emits."""
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
    return TabularDocument(tables=(Table(name="predictions", columns=columns, rows=tuple(rows)),))


# Three docs, mixed positive/negative paragraphs.
_FIXTURE_ROWS = [
    (0, "#/body/0", "doc://A", 1, "#/sec1", "Indemnification", "arbitration", 0.91, True),
    (1, "#/body/1", "doc://A", 1, "#/sec1", "Indemnification", "other", 0.42, False),
    (2, "#/body/2", "doc://A", 2, "#/sec2", "Governing Law", "other", 0.10, False),
    (3, "#/body/3", "doc://B", 1, None, None, "arbitration", 0.88, True),
    (4, "#/body/4", "doc://B", 1, None, None, "arbitration", 0.85, True),
    (5, "#/body/5", "doc://C", 1, None, None, "other", 0.20, False),
    (6, "#/body/6", "doc://C", 1, None, None, "other", 0.15, False),
]


class TestAggregateMethods:
    def test_any_marks_doc_with_any_positive(self):
        agg = aggregate_predictions(
            _build_predictions(_FIXTURE_ROWS),
            by="doc_uri",
            method="any",
            positive_class="arbitration",
        )
        rows = agg.tables[0].rows
        # doc_uri (idx 0), aggregate (idx 6, last but one).
        by_doc = {r[0]: r[6] for r in rows}
        assert by_doc["doc://A"] is True  # one positive
        assert by_doc["doc://B"] is True  # both positive
        assert by_doc["doc://C"] is False  # zero positive

    def test_all_only_when_every_row_positive(self):
        agg = aggregate_predictions(
            _build_predictions(_FIXTURE_ROWS),
            by="doc_uri",
            method="all",
            positive_class="arbitration",
        )
        by_doc = {r[0]: r[6] for r in agg.tables[0].rows}
        assert by_doc["doc://A"] is False  # 1/3
        assert by_doc["doc://B"] is True  # 2/2
        assert by_doc["doc://C"] is False  # 0/2

    def test_count_returns_int(self):
        agg = aggregate_predictions(
            _build_predictions(_FIXTURE_ROWS),
            by="doc_uri",
            method="count",
            positive_class="arbitration",
        )
        by_doc = {r[0]: r[6] for r in agg.tables[0].rows}
        assert by_doc["doc://A"] == 1
        assert by_doc["doc://B"] == 2
        assert by_doc["doc://C"] == 0

    def test_max_returns_float(self):
        agg = aggregate_predictions(
            _build_predictions(_FIXTURE_ROWS),
            by="doc_uri",
            method="max",
        )
        by_doc = {r[0]: r[6] for r in agg.tables[0].rows}
        assert abs(by_doc["doc://A"] - 0.91) < 1e-6
        assert abs(by_doc["doc://B"] - 0.88) < 1e-6
        assert abs(by_doc["doc://C"] - 0.20) < 1e-6

    def test_mean_returns_float(self):
        agg = aggregate_predictions(
            _build_predictions(_FIXTURE_ROWS),
            by="doc_uri",
            method="mean",
        )
        by_doc = {r[0]: r[6] for r in agg.tables[0].rows}
        # doc://A scores: 0.91, 0.42, 0.10 → mean ≈ 0.477
        assert abs(by_doc["doc://A"] - (0.91 + 0.42 + 0.10) / 3) < 1e-6

    def test_majority(self):
        # Two docs: one with majority arbitration, one without.
        rows = [
            (0, "#/0", "doc://X", 1, None, None, "arbitration", 0.9, True),
            (1, "#/1", "doc://X", 1, None, None, "arbitration", 0.8, True),
            (2, "#/2", "doc://X", 1, None, None, "other", 0.1, False),
            (3, "#/3", "doc://Y", 1, None, None, "arbitration", 0.9, True),
            (4, "#/4", "doc://Y", 1, None, None, "other", 0.1, False),
            (5, "#/5", "doc://Y", 1, None, None, "other", 0.1, False),
        ]
        agg = aggregate_predictions(
            _build_predictions(rows),
            by="doc_uri",
            method="majority",
            positive_class="arbitration",
        )
        by_doc = {r[0]: r[6] for r in agg.tables[0].rows}
        assert by_doc["doc://X"] is True  # 2/3 > half
        assert by_doc["doc://Y"] is False  # 1/3 NOT > half


class TestAggregateInputValidation:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match=r"Unknown aggregate method"):
            aggregate_predictions(
                _build_predictions(_FIXTURE_ROWS),
                method="invalid_method",  # ty: ignore[invalid-argument-type]
            )

    def test_missing_columns_raises(self):
        cols = (Column(name="other_col", column_type=ColumnType.TEXT),)
        bad = TabularDocument(tables=(Table(name="bad", columns=cols, rows=(("foo",),)),))
        with pytest.raises(ValueError, match=r"missing required columns"):
            aggregate_predictions(bad, by="other_col")

    def test_empty_table_raises(self):
        cols = _build_predictions([]).tables[0].columns
        empty = TabularDocument(tables=(Table(name="empty", columns=cols, rows=()),))
        with pytest.raises(ValueError, match=r"zero rows"):
            aggregate_predictions(empty)

    def test_no_tables_raises(self):
        empty = TabularDocument(tables=())
        with pytest.raises(ValueError, match=r"no tables"):
            aggregate_predictions(empty)


class TestAggregateSupportingBlockRefs:
    def test_supporting_block_refs_for_positive_methods(self):
        agg = aggregate_predictions(
            _build_predictions(_FIXTURE_ROWS),
            by="doc_uri",
            method="any",
            positive_class="arbitration",
        )
        # Find doc://A's row and check the supporting_block_refs (col 7).
        by_doc = {r[0]: r[7] for r in agg.tables[0].rows}
        # doc://A had one positive paragraph (#/body/0).
        assert "#/body/0" in by_doc["doc://A"]
        # doc://B had two positive paragraphs.
        assert "#/body/3" in by_doc["doc://B"]
        assert "#/body/4" in by_doc["doc://B"]


class TestAggregateBySection:
    def test_by_section_title_groups_correctly(self):
        agg = aggregate_predictions(
            _build_predictions(_FIXTURE_ROWS),
            by="section_title",
            method="any",
            positive_class="arbitration",
        )
        # Should have rows for "Indemnification", "Governing Law", and None.
        by_sec = {r[0]: r[6] for r in agg.tables[0].rows}
        assert "Indemnification" in by_sec
        assert by_sec["Indemnification"] is True  # has the arbitration row
        assert by_sec["Governing Law"] is False
