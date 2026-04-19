"""CorpusIndex — persistable compound wrapping a Corpus + retriever cache + tombstones.

Introduced in WS-3.4 per ``docs/design/fundamentals-roadmap.md`` §WS-3.
The audit at ``docs/design/corpus-actual-state.md`` showed that every
primitive needed for a unified "index" artifact already ships:

- ``Corpus.retriever(method)`` already lazy-builds and caches BM25,
  embedding, and hybrid retrievers per ``(method, group_by, kwargs)``.
- ``Corpus.embed(cache_dir=...)`` already persists dense embeddings as
  content-hash-keyed ``.npy`` files.
- ``kaos_nlp_core.structures.InvertedIndex.save/load`` ships bincode
  serialization for the BM25 backing store.

What was missing was a single serializable object bundling **corpus +
retriever cache + reranker + tombstones + manifest** that an agent can
save once, load across requests, and query through a single
``retrieve(query, top_k, method=...)`` entrypoint.

## Persistence layout

``CorpusIndex.save(path)`` writes a directory with:

- ``manifest.json`` — :class:`CorpusIndexManifest` (dates, counts,
  hashes, the ``kaos_ml_core`` version that created it).
- ``units.jsonl`` — one :class:`CorpusUnit` per line. Drives corpus
  reconstruction on ``load`` (the BM25 index rebuilds in <1s for 10k
  passages; the expensive artifact is the dense matrix).
- ``dense.npy`` — optional precomputed embedding matrix. Present iff
  the index was saved after a call to ``retriever('embedding')`` or
  ``retriever('hybrid')``. Skipped silently otherwise.
- ``tombstones.json`` — sorted list of row ids soft-deleted via
  :meth:`remove_passage`. Filtered at retrieve-time.

``CorpusIndex.load(path)`` reads ``manifest.json`` first (cheap,
validates directory shape), then replays ``units.jsonl`` through
``Corpus.__init__``. If ``dense.npy`` exists, it's threaded into the
corpus embedding cache so the first ``retrieve(method="embedding")``
call does not re-embed.

## Deletion via tombstones

``PyInvertedIndex`` has no ``remove_document`` in its Rust binding
(audit §2.2). A full rebuild on every delete is O(N) — unacceptable for
any corpus that grows over time. ``CorpusIndex`` instead tracks a
tombstone set of row ids; ``retrieve()`` filters matching hits out
before returning top-k. ``add_passages()`` is additive and returns a
new ``CorpusIndex`` because the underlying ``Corpus.extend()`` is
immutable.

## What's deliberately NOT here (yet)

- **VFS persistence.** ``save()``/``load()`` use filesystem paths today,
  mirroring ``Corpus.embed(cache_dir=...)``. A ``save_vfs``/``load_vfs``
  pair can land in a follow-on PR once a consumer (ResearchAgent) needs
  it — the VFS API is async byte-oriented and adds complexity we don't
  need yet.
- **Incremental BM25 updates.** ``PyInvertedIndex.add_document`` exists
  but the Python ``Searcher`` wrapper does not expose it as a hot add
  path. ``add_passages`` triggers a full retriever cache invalidation
  + lazy rebuild on next ``retrieve``. Revisit if profiling shows
  rebuild time matters.
- **Multi-method composition.** Each ``retrieve()`` call picks exactly
  one method. Fan-out-and-fuse is already in
  ``kaos_nlp_core.retrieval.hybrid.HybridRetriever`` — pass
  ``method="hybrid"`` to use it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import pathlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_ml_core.corpus import Corpus, CorpusUnit
from kaos_ml_core.errors import CorpusError

if TYPE_CHECKING:
    import numpy as np
    from kaos_nlp_core.retrieval.protocol import SearchHit

logger = get_logger(__name__)

_MANIFEST_FILENAME = "manifest.json"
_UNITS_FILENAME = "units.jsonl"
_DENSE_FILENAME = "dense.npy"
_TOMBSTONES_FILENAME = "tombstones.json"


@dataclass(frozen=True, slots=True)
class CorpusIndexManifest:
    """Manifest header written to ``manifest.json`` alongside the blobs.

    Captures enough provenance for a future loader to (a) refuse an
    incompatible version, (b) detect corpus drift, (c) decide whether
    ``dense.npy`` is still usable against the current embedding model.
    """

    version: int
    """Manifest schema version. Bump on breaking changes."""

    created_at: str
    """ISO 8601 UTC timestamp of the save call."""

    corpus_hash: str
    """16-hex-char sha256 prefix over the sorted (doc_uri, block_ref, text)
    tuples. Detects corpus drift across save/load."""

    unit_count: int
    """Number of units when saved (pre-tombstones)."""

    tombstone_count: int
    """Number of tombstoned units when saved."""

    embed_model: str | None
    """Identifier of the embedding model used to produce ``dense.npy``,
    or ``None`` if no dense matrix was saved."""

    dense_shape: tuple[int, int] | None
    """Shape of ``dense.npy`` when present."""

    kaos_ml_core_version: str
    """Version of kaos-ml-core at save time — reject hard version drift on load."""


# Accumulator / mutable builder. Kept separate from the frozen manifest so
# the manifest is a point-in-time snapshot, not a live view. The
# ``tombstones`` set is grown by :meth:`CorpusIndex.remove_passage`; there
# is no other mutation surface, so this stays mutable-by-design.
@dataclass(slots=True)
class _IndexState:
    tombstones: set[int] = field(default_factory=set)


class CorpusIndex:
    """Unified retrieval surface over a :class:`Corpus` with persistence.

    A thin orchestrator — the heavy lifting (BM25, embeddings, hybrid)
    stays in ``Corpus.retriever(method)``. ``CorpusIndex`` owns:

    - A reference to the underlying :class:`Corpus`.
    - A set of soft-deleted row ids (tombstones).
    - An optional reranker.
    - The save/load serialization contract.

    ``retrieve(query, top_k, method)`` dispatches to
    ``corpus.retriever(method)``, applies tombstone filtering, and
    optionally reranks.
    """

    __slots__ = ("_corpus", "_reranker", "_state")

    def __init__(
        self,
        corpus: Corpus,
        *,
        reranker: Any = None,
        tombstones: Iterable[int] | None = None,
    ) -> None:
        self._corpus = corpus
        self._reranker = reranker
        self._state = _IndexState(tombstones=set(tombstones) if tombstones else set())

    # ── Accessors ──────────────────────────────────────────────────────

    @property
    def corpus(self) -> Corpus:
        """The wrapped :class:`Corpus`. Immutable — use ``add_passages``
        to extend or ``remove_passage`` to tombstone."""
        return self._corpus

    @property
    def tombstones(self) -> frozenset[int]:
        """Read-only view of the tombstone set."""
        return frozenset(self._state.tombstones)

    @property
    def size(self) -> int:
        """Live passage count: ``corpus.size - len(tombstones)``."""
        return len(self._corpus) - len(self._state.tombstones)

    # ── Mutation ───────────────────────────────────────────────────────

    def remove_passage(self, row: int) -> None:
        """Soft-delete the passage at ``row``.

        Filtered out of every subsequent ``retrieve`` call. Does NOT
        rebuild any retriever. A full rebuild only happens on explicit
        :meth:`compact` or when ``add_passages`` creates a fresh Corpus.
        """
        if row < 0 or row >= len(self._corpus):
            msg = (
                f"row {row} out of range [0, {len(self._corpus)}). "
                "Fix: pass a row returned by corpus.iter_passages(). "
                "Alternative: use corpus.row_for(block_ref) to resolve a block_ref to its row."
            )
            raise IndexError(msg)
        self._state.tombstones.add(row)

    def add_passages(self, documents: Iterable[Any]) -> CorpusIndex:
        """Append documents and return a NEW CorpusIndex.

        Underlying ``Corpus.extend()`` is immutable (returns a new
        ``Corpus`` without retriever/embedding caches). The returned
        index inherits this instance's tombstones — the tombstone row
        ids remain valid because ``Corpus.extend`` preserves existing
        row indices and appends new ones.
        """
        new_corpus = self._corpus.extend(list(documents))
        return CorpusIndex(
            new_corpus,
            reranker=self._reranker,
            tombstones=self._state.tombstones,
        )

    # ── Retrieval ──────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        method: str = "bm25",
        group_by: str | None = None,
        over_retrieve: int | None = None,
        **retriever_kwargs: Any,
    ) -> list[SearchHit]:
        """Retrieve the top-k hits, applying tombstone filtering + optional rerank.

        Args:
            query: Natural-language query.
            top_k: Number of hits to return after tombstone filtering
                and reranking.
            method: ``"bm25"`` | ``"embedding"`` | ``"hybrid"``.
                Delegates to :meth:`Corpus.retriever`.
            group_by: Optional attribute name for coarse-grained
                retrieval (see ``Corpus.retriever``).
            over_retrieve: Fetch this many hits from the underlying
                retriever before tombstone filtering + reranking. Default
                is ``top_k + 2 * len(tombstones)`` capped at ``corpus.size``
                — enough slack that a handful of tombstones does not
                starve the final top_k.
            **retriever_kwargs: Forwarded to ``Corpus.retriever``.

        Returns:
            Up to ``top_k`` ``SearchHit`` results, tombstone-filtered and
            optionally reranked.
        """
        retriever = self._corpus.retriever(method, group_by=group_by, **retriever_kwargs)
        if over_retrieve is None:
            over_retrieve = min(
                top_k + 2 * max(len(self._state.tombstones), 1),
                max(len(self._corpus), top_k),
            )
        hits: list[SearchHit] = await retriever.retrieve(query, top_k=over_retrieve)

        if self._state.tombstones:
            hits = [h for h in hits if _hit_row_id(h) not in self._state.tombstones]

        hits = hits[:top_k]

        if self._reranker is not None and hits:
            from kaos_nlp_core.retrieval.reranker import RankedResult

            ranked: list[RankedResult] = await self._reranker.rerank(query, hits, top_k=top_k)
            # Repackage RankedResult back into the hit shape retrievers return;
            # this keeps retrieve()'s return type stable across rerank on/off.
            hits = [r.hit for r in ranked]  # type: ignore[attr-defined]

        return hits

    # ── Persistence ────────────────────────────────────────────────────

    def save(
        self, path: str | pathlib.Path, *, embed_model: str | None = None
    ) -> CorpusIndexManifest:
        """Write the index to ``path`` (created if absent).

        Files written:
        - ``manifest.json``
        - ``units.jsonl``
        - ``tombstones.json``
        - ``dense.npy`` — only if ``embed_model`` is passed AND the
          corpus embedding cache for that model is populated.

        Args:
            path: Directory to write into. Created if it does not exist.
            embed_model: If set, look up the precomputed embedding
                matrix in ``corpus._embedding_cache`` and write it as
                ``dense.npy``. Omit to save BM25 artifacts only.

        Returns:
            The :class:`CorpusIndexManifest` that was written.
        """
        out = pathlib.Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # Units
        units_path = out / _UNITS_FILENAME
        with units_path.open("w", encoding="utf-8") as handle:
            for unit in self._corpus:
                handle.write(json.dumps(dataclasses.asdict(unit)) + "\n")

        # Tombstones
        tombstones_path = out / _TOMBSTONES_FILENAME
        tombstones_path.write_text(json.dumps(sorted(self._state.tombstones)))

        # Optional dense
        dense_shape: tuple[int, int] | None = None
        if embed_model is not None:
            matrix = self._find_embedding(embed_model)
            if matrix is not None:
                import numpy as np

                np.save(out / _DENSE_FILENAME, matrix)
                dense_shape = (int(matrix.shape[0]), int(matrix.shape[1]))
                logger.info(
                    "CorpusIndex.save: wrote dense matrix shape=%s for model=%r",
                    dense_shape,
                    embed_model,
                )
            else:
                logger.warning(
                    "CorpusIndex.save: embed_model=%r requested but no cached "
                    "embedding matrix found; skipping dense.npy",
                    embed_model,
                )

        manifest = CorpusIndexManifest(
            version=1,
            created_at=datetime.now(tz=UTC).isoformat(),
            corpus_hash=_hash_corpus(self._corpus),
            unit_count=len(self._corpus),
            tombstone_count=len(self._state.tombstones),
            embed_model=embed_model if dense_shape is not None else None,
            dense_shape=dense_shape,
            kaos_ml_core_version=_kaos_ml_core_version(),
        )
        (out / _MANIFEST_FILENAME).write_text(json.dumps(dataclasses.asdict(manifest), indent=2))
        return manifest

    @classmethod
    def load(cls, path: str | pathlib.Path) -> CorpusIndex:
        """Reconstruct an index written by :meth:`save`.

        Validates the manifest's ``kaos_ml_core_version`` matches the
        installed package; a mismatch logs a warning rather than
        raising. Drift on corpus_hash is tolerated silently — the hash
        is diagnostic, not a guard.

        Args:
            path: Directory previously passed to :meth:`save`.

        Returns:
            A :class:`CorpusIndex` whose corpus is reconstructed from
            ``units.jsonl``. If ``dense.npy`` is present, the matrix is
            injected into ``corpus._embedding_cache`` under the manifest's
            ``embed_model`` key so the first
            ``retrieve(method="embedding")`` call does not re-embed.
        """
        in_dir = pathlib.Path(path)
        manifest_path = in_dir / _MANIFEST_FILENAME
        if not manifest_path.is_file():
            msg = (
                f"CorpusIndex.load: missing {_MANIFEST_FILENAME} at {in_dir}. "
                "Fix: pass the directory that was previously returned by CorpusIndex.save(). "
                "Alternative: rebuild with CorpusIndex(Corpus.from_documents([...]))."
            )
            raise CorpusError(msg)

        raw_manifest = json.loads(manifest_path.read_text())
        dense_shape_raw = raw_manifest.get("dense_shape")
        manifest = CorpusIndexManifest(
            version=int(raw_manifest["version"]),
            created_at=str(raw_manifest["created_at"]),
            corpus_hash=str(raw_manifest["corpus_hash"]),
            unit_count=int(raw_manifest["unit_count"]),
            tombstone_count=int(raw_manifest["tombstone_count"]),
            embed_model=raw_manifest.get("embed_model"),
            dense_shape=tuple(dense_shape_raw) if dense_shape_raw else None,  # type: ignore[arg-type]
            kaos_ml_core_version=str(raw_manifest["kaos_ml_core_version"]),
        )
        if manifest.version != 1:
            msg = (
                f"CorpusIndex.load: manifest version {manifest.version} not "
                "supported by this build (expected 1). "
                "Fix: re-save the index with the current kaos-ml-core. "
                "Alternative: pin kaos-ml-core to the version that wrote this manifest."
            )
            raise CorpusError(msg)
        installed = _kaos_ml_core_version()
        if manifest.kaos_ml_core_version != installed:
            logger.warning(
                "CorpusIndex.load: manifest written by kaos-ml-core %s, "
                "currently running %s. Proceeding — bincode blobs are still "
                "bincode, but behavior may differ.",
                manifest.kaos_ml_core_version,
                installed,
            )

        # Units
        units: list[CorpusUnit] = []
        with (in_dir / _UNITS_FILENAME).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                units.append(CorpusUnit(**json.loads(line)))
        corpus = Corpus(units, corpus_metadata={"reloaded": True, "unit_count": len(units)})

        # Optional dense — inject into the corpus embedding cache so the first
        # retriever(method='embedding') call does not re-embed.
        if manifest.embed_model is not None and (in_dir / _DENSE_FILENAME).is_file():
            import numpy as np

            matrix: np.ndarray = np.load(in_dir / _DENSE_FILENAME)  # type: ignore[no-untyped-call]
            # The key shape matches Corpus.embed's internal cache key
            # (see corpus.py:_embedding_cache usage) — a (model, batch_size)
            # tuple. Use batch_size=32 to match the default.
            corpus._embedding_cache[(manifest.embed_model, 32)] = matrix

        # Tombstones
        tombstones: set[int] = set()
        tombstones_path = in_dir / _TOMBSTONES_FILENAME
        if tombstones_path.is_file():
            tombstones = set(json.loads(tombstones_path.read_text()))

        return cls(corpus, tombstones=tombstones)

    # ── Internals ──────────────────────────────────────────────────────

    def _find_embedding(self, embed_model: str) -> np.ndarray | None:
        """Look up a cached embedding matrix on the underlying corpus.

        ``Corpus.embed`` keys its cache by ``(model, batch_size)``. We
        accept any batch_size for the requested model.
        """
        for (model, _batch_size), matrix in self._corpus._embedding_cache.items():
            if model == embed_model:
                return matrix
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_corpus(corpus: Corpus) -> str:
    """16-hex-char sha256 prefix over a stable projection of the corpus.

    Used as a diagnostic tag in the manifest — lets a human eyeball
    whether two saved indices are over the same corpus. Not a
    tamper-evident integrity check.
    """
    hasher = hashlib.sha256()
    for unit in corpus:
        hasher.update(unit.doc_uri.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(unit.block_ref.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(unit.text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def _kaos_ml_core_version() -> str:
    """Version of the installed kaos-ml-core package."""
    try:
        return importlib.metadata.version("kaos-ml-core")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _hit_row_id(hit: Any) -> int:
    """Extract a stable row id from any retriever's hit shape.

    Retrievers built via ``Corpus.retriever`` carry the ``CorpusUnit.row``
    as ``hit.doc_id`` (Searcher's ``id_field="id"`` is mapped from
    ``row``). Fall through to the ``external_id``-encoded row if needed.
    """
    doc_id = getattr(hit, "doc_id", None)
    if isinstance(doc_id, int):
        return doc_id
    # Defensive: some older hit shapes stringify doc_id.
    if isinstance(doc_id, str) and doc_id.isdigit():
        return int(doc_id)
    return -1  # sentinel — never matches a real tombstone, so hit passes through


__all__ = ["CorpusIndex", "CorpusIndexManifest"]
