"""Corpus — AST-grounded row/block_ref bidirectional map.

This module is the heart of kaos-ml-core. It implements the five
AST-grounding invariants from ``docs/internal/prd/kaos-ml-core.md`` §5:

1. ``corpus.unit(r).row == r``
2. ``corpus.row_for(corpus.unit(r).block_ref) == r`` (first matching row
   for sentence-level corpora; use ``rows_for`` for the full set)
3. ``corpus.block_ref_for(r) == corpus.unit(r).block_ref``
4. Any feature matrix ``X`` produced from this Corpus has
   ``X.shape[0] == len(corpus)`` and row ``i`` is the featurization of
   ``corpus.unit(i).text``.
5. Predictions emitted via ``predict_corpus(corpus, X, clf)`` are a
   ``TabularDocument`` with one row per ``CorpusUnit``, joined by row
   index, carrying the ``block_ref`` for every prediction.

The pattern mirrors ``kaos_content.search._paragraphs_to_records`` and
``kaos_nlp_core.search.Searcher`` — internal int row indices are bound
to AST ``block_ref``s by position in the units list, and round-trip
through both ``row_for`` and ``block_ref_for``. The actual paragraph and
sentence enumeration is delegated to ``kaos_content.units`` so there is
exactly one source of truth for "what counts as a row."
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kaos_ml_core.errors import CorpusError

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument


@dataclass(frozen=True, slots=True)
class CorpusUnit:
    """A single AST-grounded unit (paragraph or sentence) in a Corpus.

    Fields exactly mirror the records dict that
    ``kaos_content.search._paragraphs_to_records`` threads through
    ``Searcher.from_documents``, plus a ``doc_uri`` for multi-document
    corpora and an explicit dense ``row`` index.
    """

    row: int
    """Internal row index (0..len(corpus)-1). Stable for the lifetime
    of the Corpus. Equal to the row index in any feature matrix X
    produced from this Corpus."""

    text: str
    """The paragraph or sentence text."""

    block_ref: str
    """JSON pointer into the AST (e.g. ``#/body/12``). For sentence-level
    units this is the ``block_ref`` of the *containing paragraph* — many
    sentences may share one paragraph block_ref."""

    doc_uri: str
    """Source ContentDocument URI. Read from
    ``document.metadata.source.uri`` or supplied explicitly via
    ``Corpus.from_documents(doc_uris=...)``."""

    page: int | None
    """1-indexed page number from provenance, or None."""

    section_ref: str | None
    """Heading ref of the containing section, or None."""

    section_title: str | None
    """Resolved heading text of the containing section, or None."""


class Corpus:
    """A frozen, AST-grounded set of text units with bidirectional mapping
    between internal row indices and AST ``block_ref``s.

    Constructed from one or more ``ContentDocument`` instances. Row
    indices are dense (0..len(corpus)-1) and stable for the lifetime of
    the Corpus. The same row index is the row index in any feature
    matrix produced from this Corpus, satisfying the AST-grounding
    invariants in PRD §5.

    Use ``Corpus.from_paragraphs(doc)`` or ``Corpus.from_sentences(doc)``
    for single-document construction; ``Corpus.from_documents(docs, ...)``
    for multi-document. The level keyword chooses paragraph- or
    sentence-level granularity.
    """

    def __init__(self, units: Sequence[CorpusUnit]) -> None:
        if not units:
            msg = (
                "Corpus is empty. "
                "Cause: the source document(s) had no non-empty paragraphs/sentences. "
                "Fix: verify the documents extracted text correctly. "
                "Alternative: try Corpus.from_documents(level='paragraph') if "
                "sentence segmentation is failing."
            )
            raise CorpusError(msg)

        # Verify dense row indices — required for the row==index invariant.
        for i, u in enumerate(units):
            if u.row != i:
                msg = (
                    f"Corpus units must have dense row indices starting at 0; "
                    f"got row={u.row} at position {i}. "
                    "Fix: build the Corpus via the Corpus.from_* class methods, "
                    "which assign row indices automatically."
                )
                raise CorpusError(msg)

        self._units: tuple[CorpusUnit, ...] = tuple(units)
        # First-row index per block_ref. For sentence-level corpora,
        # multiple sentences share one paragraph block_ref; row_for
        # returns the first; rows_for returns all.
        self._first_row: dict[str, int] = {}
        self._all_rows: dict[str, list[int]] = {}
        for u in self._units:
            self._first_row.setdefault(u.block_ref, u.row)
            self._all_rows.setdefault(u.block_ref, []).append(u.row)

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def from_paragraphs(
        cls,
        document: ContentDocument,
        *,
        doc_uri: str | None = None,
    ) -> Corpus:
        """Build a paragraph-level Corpus from a single ContentDocument."""
        return cls.from_documents(
            [document],
            level="paragraph",
            doc_uris=None if doc_uri is None else [doc_uri],
        )

    @classmethod
    def from_sentences(
        cls,
        document: ContentDocument,
        *,
        doc_uri: str | None = None,
    ) -> Corpus:
        """Build a sentence-level Corpus from a single ContentDocument.

        Requires kaos-nlp-core for sentence segmentation. Each sentence
        carries the ``block_ref`` of its containing paragraph.
        """
        return cls.from_documents(
            [document],
            level="sentence",
            doc_uris=None if doc_uri is None else [doc_uri],
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[ContentDocument],
        *,
        level: str = "paragraph",
        doc_uris: Iterable[str | None] | None = None,
    ) -> Corpus:
        """Build a Corpus from one or more ContentDocuments.

        Args:
            documents: ContentDocument iterables.
            level: ``"paragraph"`` (default) or ``"sentence"``.
            doc_uris: Optional per-document URI overrides. When ``None``,
                the URI is read from ``document.metadata.source.uri``.
                If neither is available, ``CorpusError`` is raised.

        Raises:
            CorpusError: On unknown level, mismatched doc_uris length,
                empty input, or missing doc URI.
        """
        from kaos_content.units import (
            iter_paragraph_units,
            iter_sentence_units,
        )

        docs = list(documents)
        if not docs:
            msg = (
                "Corpus.from_documents requires at least one ContentDocument. "
                "Fix: pass a non-empty iterable of documents."
            )
            raise CorpusError(msg)

        uris: list[str | None] = list(doc_uris) if doc_uris is not None else [None] * len(docs)
        if len(uris) != len(docs):
            msg = (
                f"doc_uris length ({len(uris)}) must match documents length ({len(docs)}). "
                "Fix: pass exactly one URI per document, or pass None to read from metadata."
            )
            raise CorpusError(msg)

        if level not in ("paragraph", "sentence"):
            msg = (
                f"Unknown level={level!r}. "
                "Fix: use 'paragraph' (default) or 'sentence'. "
                "Sentence-level requires the kaos-nlp-core sentence segmenter."
            )
            raise CorpusError(msg)

        units: list[CorpusUnit] = []
        global_row = 0

        for doc, override_uri in zip(docs, uris, strict=True):
            doc_uri = override_uri if override_uri is not None else _resolve_doc_uri(doc)

            local_units = (
                iter_paragraph_units(doc) if level == "paragraph" else iter_sentence_units(doc)
            )

            for lu in local_units:
                units.append(
                    CorpusUnit(
                        row=global_row,
                        text=lu.text,
                        block_ref=lu.block_ref,
                        doc_uri=doc_uri,
                        page=lu.page,
                        section_ref=lu.section_ref,
                        section_title=lu.section_title,
                    )
                )
                global_row += 1

        return cls(units)

    # ── Bidirectional row ↔ block_ref mapping ──────────────────────────

    def __len__(self) -> int:
        return len(self._units)

    def __iter__(self) -> Iterator[CorpusUnit]:
        return iter(self._units)

    @property
    def units(self) -> tuple[CorpusUnit, ...]:
        return self._units

    def unit(self, row: int) -> CorpusUnit:
        """Return the CorpusUnit at the given row index."""
        if row < 0 or row >= len(self._units):
            msg = f"row index {row} out of range [0, {len(self._units)})"
            raise IndexError(msg)
        return self._units[row]

    def block_ref_for(self, row: int) -> str:
        """Return the block_ref for the given row index."""
        return self.unit(row).block_ref

    def row_for(self, block_ref: str) -> int:
        """Return the first row whose unit has the given block_ref.

        For sentence-level corpora multiple rows share a paragraph
        block_ref; use ``rows_for()`` to retrieve all of them.

        Raises:
            KeyError: If no row has the given block_ref.
        """
        if block_ref not in self._first_row:
            msg = (
                f"block_ref {block_ref!r} not found in Corpus. "
                "Fix: verify the block_ref came from a Corpus built over the same document set. "
                "Alternative: use rows_for() if you expect multiple rows per ref."
            )
            raise KeyError(msg)
        return self._first_row[block_ref]

    def rows_for(self, block_ref: str) -> list[int]:
        """Return all row indices whose units have the given block_ref.

        Returns an empty list if no row matches.
        """
        return list(self._all_rows.get(block_ref, ()))

    def units_for_doc(self, doc_uri: str) -> list[int]:
        """Return all row indices whose units came from the given doc URI."""
        return [u.row for u in self._units if u.doc_uri == doc_uri]

    # ── Embedding cache ────────────────────────────────────────────────

    _embeddings: Any = None  # np.ndarray | None, typed as Any to avoid numpy import

    def embed(
        self,
        *,
        model: str | None = None,
        batch_size: int = 32,
    ) -> Any:
        """Return a dense embedding matrix for this Corpus, caching the result.

        Uses ``kaos_ml_core.features.embed_corpus`` on first call and
        caches the result so subsequent calls (or
        ``EmbeddingRetriever.from_corpus``) can reuse the embeddings
        without re-running the model.

        Args:
            model: Embedding model id.  Defaults to the registry
                default (``BAAI/bge-small-en-v1.5``).
            batch_size: Inference batch size.

        Returns:
            A float32 numpy array of shape ``(len(self), dim)``.
            Row ``i`` is the embedding for ``self.unit(i).text``.
        """
        if self._embeddings is None:
            from kaos_ml_core.features import embed_corpus

            self._embeddings = embed_corpus(self, model=model, batch_size=batch_size)
        return self._embeddings

    # ── Retriever factory (multi-level, cached) ────────────────────────

    _retriever_cache: dict[tuple[str, str | None], Any] | None = None

    def retriever(
        self,
        method: str = "bm25",
        *,
        group_by: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get a ``Retriever`` for this Corpus, cached by (method, group_by).

        Supports coarse-to-fine retrieval: build a section-level
        retriever AND a paragraph-level retriever from the same Corpus
        without rebuilding or re-embedding.

        Args:
            method: ``"bm25"``, ``"embedding"``, or ``"hybrid"``.
            group_by: Optional attribute name on ``CorpusUnit`` to
                group by before indexing (e.g. ``"section_ref"`` for
                section-level retrieval).  ``None`` gives paragraph-
                or sentence-level (whatever level the Corpus was built
                with).
            **kwargs: Forwarded to the retriever's ``from_corpus()``.

        Returns:
            A ``Retriever`` protocol instance.  Cached: calling
            ``corpus.retriever("bm25")`` twice returns the same object.

        Example — coarse-to-fine::

            section_r = corpus.retriever("embedding", group_by="section_ref")
            paragraph_r = corpus.retriever("embedding")
            # Use section_r for broad queries, paragraph_r for drill-down.

        Raises:
            ValueError: On unknown method.
            ImportError: If the required package is not installed.
        """
        if self._retriever_cache is None:
            self._retriever_cache = {}

        cache_key = (method, group_by)
        if cache_key in self._retriever_cache:
            return self._retriever_cache[cache_key]

        if method == "bm25":
            from kaos_nlp_core.retrieval.bm25 import BM25Retriever

            ret = BM25Retriever.from_corpus(self, group_by=group_by, **kwargs)
        elif method == "embedding":
            try:
                from kaos_nlp_transformers.retrieval import EmbeddingRetriever
            except ImportError as exc:
                msg = (
                    "Corpus.retriever('embedding') requires kaos-nlp-transformers. "
                    "Fix: install kaos-nlp-transformers, or use method='bm25'."
                )
                raise ImportError(msg) from exc
            ret = EmbeddingRetriever.from_corpus(self, group_by=group_by, **kwargs)
        elif method == "hybrid":
            from kaos_nlp_core.retrieval.hybrid import HybridRetriever

            ret = HybridRetriever.from_corpus(self, group_by=group_by, **kwargs)
        else:
            msg = (
                f"Unknown retriever method {method!r}. "
                "Fix: use 'bm25', 'embedding', or 'hybrid'."
            )
            raise ValueError(msg)

        self._retriever_cache[cache_key] = ret
        return ret

    # ── TabularDocument bridge ─────────────────────────────────────────

    def to_tabular(self):
        """Return a TabularDocument with one row per CorpusUnit.

        Drops straight into kaos-tabular for SQL queries and into
        kaos-mcp for resource templates. Free.
        """
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
            Column(name="text", column_type=ColumnType.TEXT),
        )
        rows = tuple(
            (
                u.row,
                u.block_ref,
                u.doc_uri,
                u.page,
                u.section_ref,
                u.section_title,
                u.text,
            )
            for u in self._units
        )
        table = Table(name="corpus", columns=columns, rows=rows)
        return TabularDocument(tables=(table,))


def _resolve_doc_uri(doc: ContentDocument) -> str:
    """Resolve a document URI from metadata, raising if absent."""
    src = doc.metadata.source
    if src is not None and src.uri:
        return src.uri
    msg = (
        "ContentDocument has no metadata.source.uri. "
        "Fix: pass doc_uris=[...] explicitly to Corpus.from_documents(), "
        "or set document.metadata.source.uri before constructing the Corpus."
    )
    raise CorpusError(msg)


__all__ = ["Corpus", "CorpusUnit"]
