"""Unit tests for :class:`kaos_ml_core.index.CorpusIndex`.

Covers:
- round-trip save/load over a real Corpus built from ContentDocuments,
- tombstone filtering at retrieve time,
- manifest schema + version guard,
- add_passages returns a new index with tombstones preserved,
- index size excludes tombstoned rows.

Live retrieval is not covered here — the BM25 backend is already
exercised by ``tests/unit/test_corpus.py`` and by the kaos-llm-core live
RAG tests. This module only verifies the CorpusIndex orchestration.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from kaos_content.model.attr import Provenance, SourceRef
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata

from kaos_ml_core.corpus import Corpus, CorpusUnit
from kaos_ml_core.errors import CorpusError
from kaos_ml_core.index import CorpusIndex, CorpusIndexManifest

_source = SourceRef(uri="doc:test-index")


def _prov(page: int = 1) -> Provenance:
    return Provenance(source=_source, page=page)


def _para(text: str, page: int = 1) -> Paragraph:
    return Paragraph(children=(Text(value=text),), provenance=_prov(page))


def _doc(title: str, paragraphs: list[str], page: int = 1) -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=title, source=SourceRef(uri=f"doc:test-{title}")),
        body=tuple(_para(p, page) for p in paragraphs),
    )


@pytest.fixture()
def sample_corpus() -> Corpus:
    doc_a = _doc("alpha", ["alpha one introduces alpha.", "alpha two expands the idea."])
    doc_b = _doc("beta", ["beta one shifts focus.", "beta two continues the argument."])
    return Corpus.from_documents([doc_a, doc_b], level="paragraph")


@pytest.mark.unit
class TestCorpusIndexBasics:
    def test_constructs_over_corpus(self, sample_corpus: Corpus) -> None:
        index = CorpusIndex(sample_corpus)
        assert index.size == len(sample_corpus) == 4
        assert index.tombstones == frozenset()

    def test_corpus_is_readonly_accessor(self, sample_corpus: Corpus) -> None:
        index = CorpusIndex(sample_corpus)
        assert index.corpus is sample_corpus


@pytest.mark.unit
class TestCorpusIndexTombstones:
    def test_remove_passage_shrinks_size(self, sample_corpus: Corpus) -> None:
        index = CorpusIndex(sample_corpus)
        index.remove_passage(2)
        assert index.size == 3
        assert 2 in index.tombstones

    def test_remove_passage_out_of_range_raises(self, sample_corpus: Corpus) -> None:
        index = CorpusIndex(sample_corpus)
        with pytest.raises(IndexError, match="out of range"):
            index.remove_passage(42)
        with pytest.raises(IndexError):
            index.remove_passage(-1)

    def test_tombstones_property_is_frozen(self, sample_corpus: Corpus) -> None:
        index = CorpusIndex(sample_corpus)
        index.remove_passage(0)
        tombstones = index.tombstones
        assert isinstance(tombstones, frozenset)
        # Mutation of the returned snapshot must not propagate.
        assert 0 in tombstones


@pytest.mark.unit
class TestCorpusIndexPersistence:
    def test_save_writes_expected_files(
        self, sample_corpus: Corpus, tmp_path: pathlib.Path
    ) -> None:
        index = CorpusIndex(sample_corpus)
        index.remove_passage(0)
        index.remove_passage(3)

        manifest = index.save(tmp_path)
        assert isinstance(manifest, CorpusIndexManifest)
        assert manifest.version == 1
        assert manifest.unit_count == 4
        assert manifest.tombstone_count == 2
        assert manifest.embed_model is None
        assert manifest.dense_shape is None

        assert (tmp_path / "manifest.json").is_file()
        assert (tmp_path / "units.jsonl").is_file()
        assert (tmp_path / "tombstones.json").is_file()
        # No dense without embed_model passed to save
        assert not (tmp_path / "dense.npy").is_file()

    def test_units_jsonl_is_one_unit_per_line(
        self, sample_corpus: Corpus, tmp_path: pathlib.Path
    ) -> None:
        CorpusIndex(sample_corpus).save(tmp_path)
        lines = (tmp_path / "units.jsonl").read_text().splitlines()
        assert len(lines) == len(sample_corpus)
        first = json.loads(lines[0])
        assert first["row"] == 0
        assert "text" in first
        assert "block_ref" in first
        assert "doc_uri" in first

    def test_round_trip_preserves_units_and_tombstones(
        self, sample_corpus: Corpus, tmp_path: pathlib.Path
    ) -> None:
        original = CorpusIndex(sample_corpus)
        original.remove_passage(1)
        original.save(tmp_path)

        loaded = CorpusIndex.load(tmp_path)
        assert loaded.size == original.size
        assert loaded.tombstones == original.tombstones
        # Every unit survived the JSONL round trip.
        for orig, reloaded in zip(list(original.corpus), list(loaded.corpus), strict=True):
            assert isinstance(reloaded, CorpusUnit)
            assert reloaded.row == orig.row
            assert reloaded.text == orig.text
            assert reloaded.block_ref == orig.block_ref
            assert reloaded.doc_uri == orig.doc_uri

    def test_load_rejects_missing_manifest(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(CorpusError, match="missing manifest"):
            CorpusIndex.load(tmp_path)

    def test_load_rejects_future_manifest_version(
        self, sample_corpus: Corpus, tmp_path: pathlib.Path
    ) -> None:
        CorpusIndex(sample_corpus).save(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        raw = json.loads(manifest_path.read_text())
        raw["version"] = 999
        manifest_path.write_text(json.dumps(raw))
        with pytest.raises(CorpusError, match="version 999"):
            CorpusIndex.load(tmp_path)

    def test_dense_matrix_is_persisted_and_reloaded(
        self, sample_corpus: Corpus, tmp_path: pathlib.Path
    ) -> None:
        # Inject a fake embedding cache entry so save() picks it up
        # without a live embedder round-trip.
        import numpy as np

        matrix = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
        sample_corpus._embedding_cache[("fake-model", 32)] = matrix
        index = CorpusIndex(sample_corpus)
        manifest = index.save(tmp_path, embed_model="fake-model")
        assert manifest.embed_model == "fake-model"
        assert manifest.dense_shape == (4, 8)
        assert (tmp_path / "dense.npy").is_file()

        loaded = CorpusIndex.load(tmp_path)
        cached = loaded.corpus._embedding_cache[("fake-model", 32)]
        np.testing.assert_array_equal(cached, matrix)


@pytest.mark.unit
class TestCorpusIndexAddPassages:
    def test_add_passages_returns_new_index(self, sample_corpus: Corpus) -> None:
        original = CorpusIndex(sample_corpus)
        new_doc = _doc("gamma", ["gamma one introduces gamma."])
        extended = original.add_passages([new_doc])

        assert extended is not original
        assert extended.size == original.size + 1
        assert original.size == 4  # original unchanged

    def test_add_passages_preserves_tombstones(self, sample_corpus: Corpus) -> None:
        original = CorpusIndex(sample_corpus)
        original.remove_passage(2)
        extended = original.add_passages([_doc("gamma", ["new paragraph."])])
        assert 2 in extended.tombstones


@pytest.mark.unit
class TestCorpusIndexRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_returns_hits(self, sample_corpus: Corpus) -> None:
        index = CorpusIndex(sample_corpus)
        hits = await index.retrieve("alpha", top_k=2)
        assert len(hits) > 0
        # BM25 should favor the paragraphs that literally say "alpha".
        assert any("alpha" in h.text.lower() for h in hits)

    @pytest.mark.asyncio
    async def test_retrieve_filters_tombstones(self, sample_corpus: Corpus) -> None:
        index = CorpusIndex(sample_corpus)
        # Tombstone every "alpha" row so the "beta" rows win even on an
        # "alpha"-leaning query.
        for row, unit in enumerate(sample_corpus):
            if "alpha" in unit.text.lower():
                index.remove_passage(row)
        hits = await index.retrieve("alpha", top_k=4)
        for hit in hits:
            assert "alpha" not in hit.text.lower(), f"tombstoned row leaked through: {hit!r}"
