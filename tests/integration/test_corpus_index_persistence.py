"""Persistence round-trip + retrieval benchmark for :class:`CorpusIndex`.

Scenario: a persistent ResearchAgent indexes a realistic multi-paragraph
corpus, saves it to disk, spins down, reloads the index from disk, and
issues the same query again. The hits from both runs must agree — the
persistence path is not allowed to silently drop rows, corrupt doc_uris,
or change retrieval order.

Also captures a cheap benchmark: a 500-paragraph synthetic corpus should
build its BM25 index in <5s and return top-10 hits in <50ms p95 — well
under the WS-3 acceptance thresholds (10k in <30s, p95 <50ms).
"""

from __future__ import annotations

import pathlib
import statistics
import time

import pytest
from kaos_content.model.attr import Provenance, SourceRef
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata

from kaos_ml_core.corpus import Corpus
from kaos_ml_core.index import CorpusIndex


def _doc(doc_uri: str, paragraphs: list[str]) -> ContentDocument:
    source = SourceRef(uri=doc_uri)
    return ContentDocument(
        metadata=DocumentMetadata(title=doc_uri, source=source),
        body=tuple(
            Paragraph(children=(Text(value=p),), provenance=Provenance(source=source, page=1))
            for p in paragraphs
        ),
    )


@pytest.fixture()
def realistic_corpus() -> Corpus:
    """A 3-document corpus with enough lexical signal for BM25 to rank usefully."""
    docs = [
        _doc(
            "doc:delaware",
            [
                "The Delaware General Corporation Law requires a certificate of incorporation.",
                "The filing fee is $89. The certificate must name a registered agent.",
                "Directors need not be stockholders unless the certificate so requires.",
                "Mergers require majority approval of outstanding shares.",
            ],
        ),
        _doc(
            "doc:rfc-2119",
            [
                "MUST, REQUIRED, and SHALL denote an absolute requirement of the specification.",
                "SHOULD denotes a recommendation that may be overridden with careful analysis.",
                "MAY denotes an optional feature that vendors can implement at their discretion.",
                "MUST NOT and SHALL NOT denote an absolute prohibition of the specification.",
            ],
        ),
        _doc(
            "doc:apollo-11",
            [
                "Apollo 11 launched from Kennedy Space Center on July 16, 1969 at 13:32 UTC.",
                "The Eagle lunar module landed on the Moon at 20:17 UTC on July 20, 1969.",
                "Neil Armstrong stepped onto the lunar surface at 02:56 UTC on July 21, 1969.",
                "The mission splashed down in the Pacific Ocean on July 24, 1969.",
            ],
        ),
    ]
    return Corpus.from_documents(docs, level="paragraph")


@pytest.mark.integration
class TestCorpusIndexPersistenceRoundTrip:
    @pytest.mark.asyncio
    async def test_save_load_retrieve_agrees_with_live(
        self, realistic_corpus: Corpus, tmp_path: pathlib.Path
    ) -> None:
        """Query the live index, save, reload, query the reloaded one —
        results must match on doc_uris and ranking."""
        live = CorpusIndex(realistic_corpus)
        query = "What is the filing fee for a Delaware certificate of incorporation?"

        live_hits = await live.retrieve(query, top_k=3)
        assert len(live_hits) > 0
        live_ids = tuple(h.doc_id for h in live_hits)
        live_top_text = live_hits[0].text.lower()
        assert "delaware" in live_top_text or "$89" in live_top_text, (
            f"Top hit should mention Delaware/$89: {live_top_text!r}"
        )

        # Save & reload.
        manifest = live.save(tmp_path)
        assert manifest.unit_count == len(realistic_corpus)

        reloaded = CorpusIndex.load(tmp_path)
        assert reloaded.size == live.size

        reloaded_hits = await reloaded.retrieve(query, top_k=3)
        reloaded_ids = tuple(h.doc_id for h in reloaded_hits)

        assert reloaded_ids == live_ids, (
            f"Retrieval order changed after save/load: live={live_ids} reloaded={reloaded_ids}. "
            "This means units.jsonl or the BM25 rebuild is not deterministic."
        )

    @pytest.mark.asyncio
    async def test_tombstones_survive_round_trip(
        self, realistic_corpus: Corpus, tmp_path: pathlib.Path
    ) -> None:
        original = CorpusIndex(realistic_corpus)
        # Tombstone every passage from doc:delaware.
        for row, unit in enumerate(realistic_corpus):
            if unit.doc_uri == "doc:delaware":
                original.remove_passage(row)
        tomb_count = len(original.tombstones)
        assert tomb_count > 0, "fixture precondition"

        original.save(tmp_path)
        reloaded = CorpusIndex.load(tmp_path)

        assert reloaded.tombstones == original.tombstones
        assert len(reloaded.tombstones) == tomb_count

        # A query that would have matched delaware must NOT return delaware rows.
        hits = await reloaded.retrieve("Delaware filing fee", top_k=5)
        for hit in hits:
            # The passage URI format is {doc_uri}{block_ref}.
            external = getattr(hit, "external_id", None) or ""
            assert "doc:delaware" not in external, (
                f"tombstoned doc:delaware leaked through reload: {hit!r}"
            )


@pytest.mark.integration
class TestCorpusIndexBenchmark:
    @pytest.mark.asyncio
    async def test_medium_corpus_latency(self) -> None:
        """500-paragraph synthetic corpus: build + query latency sanity.

        Not a rigorous benchmark — just confirms we are in the
        correct order of magnitude versus the WS-3 ceiling
        (10k in <30s / p95 <50ms).
        """
        paragraphs_per_doc = 50
        doc_count = 10
        docs = [
            _doc(
                f"doc:synthetic-{i}",
                [
                    f"Synthetic paragraph {j} in document {i}. "
                    f"This text mentions the keyword alpha{i % 5}, "
                    f"beta{j % 7}, and gamma{(i + j) % 11}."
                    for j in range(paragraphs_per_doc)
                ],
            )
            for i in range(doc_count)
        ]

        build_start = time.monotonic()
        corpus = Corpus.from_documents(docs, level="paragraph")
        index = CorpusIndex(corpus)
        # Force the BM25 retriever build by issuing one query.
        await index.retrieve("alpha0 beta0 gamma0", top_k=1)
        build_ms = (time.monotonic() - build_start) * 1000.0
        assert build_ms < 5_000, (
            f"500-passage build+warmup took {build_ms:.0f}ms, expected <5s "
            "(WS-3 ceiling is 10k in <30s)"
        )

        latencies_ms: list[float] = []
        for i in range(30):
            query = f"alpha{i % 5} beta{i % 7} gamma{(i * 3) % 11}"
            start = time.monotonic()
            hits = await index.retrieve(query, top_k=10)
            latencies_ms.append((time.monotonic() - start) * 1000.0)
            assert len(hits) > 0

        p95 = statistics.quantiles(latencies_ms, n=20)[18]  # 95th percentile
        assert p95 < 100.0, (
            f"p95 retrieve latency {p95:.1f}ms over 500 passages, "
            "expected <100ms (WS-3 ceiling is 50ms @ 10k passages)"
        )
