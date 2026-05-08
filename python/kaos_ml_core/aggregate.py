"""Cross-granularity prediction aggregation.

Bridges fine-grained predictions (paragraph / sentence / clause) up to
coarser keys (section / document) — the central operation in due-diligence
and contract-analytics use cases. Without this, "predict at clause level,
decide at doc level" is unsupported.

Builds on top of :mod:`kaos_content.model.tabular` (the canonical
``TabularDocument`` carrier) and operates by-row in pure Python — kept
out of polars/duckdb-land so the module stays dependency-light. We can
tier-up to polars / duckdb later if profiling shows it matters; in
practice these aggregations run on prediction tables of <10⁶ rows where
Python loops are bounded by milliseconds.

Downstream use cases:

- **Contract analytics**: clause-level "binding arbitration" predictions
  → aggregate to ``doc_uri`` with ``method="any"`` → list of contracts
  containing arbitration language.
- **Due diligence**: paragraph-level "indemnification" predictions →
  aggregate to ``section_ref`` with ``method="any"`` → list of indemnity
  sections, then aggregate up again to ``doc_uri`` to get per-doc
  indemnity coverage.
- **TAR**: paragraph-level responsiveness predictions → aggregate to
  ``doc_uri`` with ``method="any"`` for production decisions, OR
  ``method="count"`` for review-prioritization queues.
- **Privilege detection**: paragraph-level privilege predictions →
  aggregate to ``doc_uri`` with ``method="any"`` (mark whole doc
  privileged if any paragraph is) AND keep the supporting block_refs
  on the output for the privilege log.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kaos_content.model.tabular import TabularDocument

__all__ = ["aggregate_predictions"]


# Every aggregation method the v0 release supports. Adding a new one
# means extending _AGGREGATORS below + the ColumnType inference.
AggregateMethod = Literal["any", "all", "max", "mean", "count", "majority"]
_VALID_METHODS: frozenset[str] = frozenset({"any", "all", "max", "mean", "count", "majority"})


def aggregate_predictions(
    predictions: TabularDocument,
    *,
    by: str = "doc_uri",
    method: AggregateMethod = "any",
    positive_class: str | None = None,
    score_column: str = "score",
    class_column: str = "predicted_label",
    above_column: str = "above_threshold",
) -> TabularDocument:
    """Aggregate fine-grained predictions to a coarser key.

    Input is the ``TabularDocument`` returned by :func:`predict_corpus` /
    :meth:`Pipeline.predict` — one row per CorpusUnit with columns
    ``row``, ``block_ref``, ``doc_uri``, ``page``, ``section_ref``,
    ``section_title``, ``predicted_label``, ``score``, ``above_threshold``.

    Output is a new ``TabularDocument`` with one row per unique value of
    ``by``, plus columns describing the aggregated outcome and a
    ``supporting_block_refs`` column (semicolon-separated) that lets a UI
    drill from "this doc has arbitration" → which paragraph triggered it.

    Args:
        predictions: TabularDocument from ``predict_corpus`` /
            ``Pipeline.predict``. Must contain at least the ``by``,
            ``score_column``, ``class_column``, and ``above_column``
            columns.
        by: Aggregation key column name. Common values: ``"doc_uri"``
            (document-level decision), ``"section_ref"`` (section-level
            decision). Any column in the input table works.
        method: How to roll fine-grained predictions up:

            - ``"any"`` (default): aggregated row is positive if ANY
              underlying row is above_threshold. The marquee TAR /
              contract-analytics aggregation.
            - ``"all"``: positive only if ALL underlying rows are
              above_threshold. Use for "is this section UNIFORMLY
              privileged?" patterns.
            - ``"max"``: max score across underlying rows.
            - ``"mean"``: mean score across underlying rows.
            - ``"count"``: integer — how many underlying rows are
              above_threshold. Use for review-queue prioritization.
            - ``"majority"``: positive if more than half of underlying
              rows are above_threshold.

        positive_class: Class label considered "positive" for the
            ``"any"`` / ``"all"`` / ``"majority"`` methods. Defaults to
            the most-frequent non-empty value in ``predicted_label``
            among rows where ``above_threshold`` is True (best-effort
            inference; pass explicitly for deterministic behavior).
        score_column: Column name containing the positive-class
            probability (default ``"score"``).
        class_column: Column name containing the predicted hard label
            (default ``"predicted_label"``).
        above_column: Column name containing the boolean threshold flag
            (default ``"above_threshold"``).

    Returns:
        ``TabularDocument`` with one table named ``"aggregated_{by}"``.

    Raises:
        ValueError: On unknown method, missing required columns, or
            empty input.
    """
    from kaos_content.model.tabular import Column, ColumnType, Table, TabularDocument

    if method not in _VALID_METHODS:
        msg = f"Unknown aggregate method {method!r}. Fix: pick one of {sorted(_VALID_METHODS)}."
        raise ValueError(msg)
    if not predictions.tables:
        msg = (
            "predictions has no tables. Fix: pass the TabularDocument "
            "returned by Pipeline.predict() / predict_corpus()."
        )
        raise ValueError(msg)
    table = predictions.tables[0]
    column_names = [c.name for c in table.columns]
    required = {by, score_column, class_column, above_column}
    missing = required - set(column_names)
    if missing:
        msg = (
            f"predictions table is missing required columns: {sorted(missing)}. "
            f"Has: {column_names}. "
            "Fix: this is the output shape of Pipeline.predict() — verify "
            "you passed the right TabularDocument and didn't drop columns."
        )
        raise ValueError(msg)
    if not table.rows:
        msg = (
            "predictions table has zero rows; cannot aggregate. Fix: ensure "
            "the corpus passed to Pipeline.predict() was non-empty."
        )
        raise ValueError(msg)

    # ── infer positive_class if not given ──────────────────────────────
    by_idx = column_names.index(by)
    score_idx = column_names.index(score_column)
    class_idx = column_names.index(class_column)
    above_idx = column_names.index(above_column)
    block_ref_idx = column_names.index("block_ref") if "block_ref" in column_names else None

    if positive_class is None:
        # Most common label among above_threshold rows.
        counts: dict[str, int] = defaultdict(int)
        for row in table.rows:
            if row[above_idx]:
                label = row[class_idx]
                if label is not None:
                    counts[str(label)] += 1
        if counts:
            positive_class = max(counts, key=lambda k: counts[k])
        else:
            # No rows above threshold; pick the first non-empty class label
            # as a safe fallback. The "any"/"all"/"majority" methods will
            # all produce False; that's the correct semantics.
            for row in table.rows:
                label = row[class_idx]
                if label:
                    positive_class = str(label)
                    break

    # ── group by key ───────────────────────────────────────────────────
    groups: dict[object, list[tuple]] = defaultdict(list)
    for row in table.rows:
        groups[row[by_idx]].append(row)

    # ── per-group aggregation ──────────────────────────────────────────
    out_rows: list[tuple] = []
    for key, rows in groups.items():
        n_total = len(rows)
        # above_threshold ∧ predicted_label == positive_class
        positive_flags = [
            bool(r[above_idx]) and (positive_class is None or str(r[class_idx]) == positive_class)
            for r in rows
        ]
        n_positive = sum(positive_flags)
        scores = [float(r[score_idx]) for r in rows]
        max_score = max(scores) if scores else 0.0
        mean_score = sum(scores) / n_total if n_total else 0.0

        if method == "any":
            agg_value: bool | int | float = bool(n_positive >= 1)
        elif method == "all":
            agg_value = bool(n_positive == n_total)
        elif method == "max":
            agg_value = max_score
        elif method == "mean":
            agg_value = mean_score
        elif method == "count":
            agg_value = int(n_positive)
        else:  # majority
            agg_value = bool(n_positive * 2 > n_total)

        # Supporting block_refs: paragraphs that voted positive (for
        # "any"/"all"/"majority"/"count") OR all underlying rows (for
        # "max"/"mean"). Capped at 50 to avoid huge cells.
        if method in {"any", "all", "majority", "count"}:
            supporting = [
                str(r[block_ref_idx]) if block_ref_idx is not None else "?"
                for r, flag in zip(rows, positive_flags, strict=True)
                if flag
            ][:50]
        else:
            supporting = [
                str(r[block_ref_idx]) if block_ref_idx is not None else "?" for r in rows
            ][:50]
        supporting_blob = "; ".join(supporting)

        out_rows.append(
            (
                key,
                n_total,
                n_positive,
                max_score,
                mean_score,
                positive_class,
                agg_value,
                supporting_blob,
            )
        )

    # ── output table ───────────────────────────────────────────────────
    # Column type for "aggregate" depends on method.
    agg_column_type = (
        ColumnType.BOOLEAN
        if method in {"any", "all", "majority"}
        else ColumnType.INTEGER
        if method == "count"
        else ColumnType.FLOAT
    )

    columns = (
        Column(name=by, column_type=_infer_column_type(by_idx, table.columns)),
        Column(name="n_rows", column_type=ColumnType.INTEGER),
        Column(name="n_positive", column_type=ColumnType.INTEGER),
        Column(name="max_score", column_type=ColumnType.FLOAT),
        Column(name="mean_score", column_type=ColumnType.FLOAT),
        Column(name="positive_class", column_type=ColumnType.TEXT),
        Column(name=f"aggregate_{method}", column_type=agg_column_type),
        Column(name="supporting_block_refs", column_type=ColumnType.TEXT),
    )
    out_table = Table(
        name=f"aggregated_{by}",
        columns=columns,
        rows=tuple(out_rows),
    )
    return TabularDocument(tables=(out_table,))


def _infer_column_type(idx: int, columns: Iterable):
    """Return the ColumnType of column at index ``idx`` from a Table's
    columns iterable. Used to preserve the by-key's typing on output.
    Falls back to ColumnType.TEXT when out of range."""
    from kaos_content.model.tabular import ColumnType

    cols = list(columns)
    return cols[idx].column_type if 0 <= idx < len(cols) else ColumnType.TEXT
