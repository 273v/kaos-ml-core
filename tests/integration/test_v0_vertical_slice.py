"""v0 vertical slice end-to-end test.

Live test: requires the [transformers] and [llm] extras installed plus
ANTHROPIC_API_KEY (or KAOS_LLM_ANTHROPIC_API_KEY) configured. The test
is skipped on systems where these prerequisites are absent — that is the
correct CI behavior per the platform's no-fake-tests rule (mocked tests
are not proof of correctness; this test exists to prove the round-trip
end-to-end against real models and real PDFs).

The single gate that this test enforces is the AST round-trip: every
prediction emitted by predict_corpus() resolves back to a real paragraph
in the source ContentDocument.
"""

from __future__ import annotations

import os
import random

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.live]


@pytest.mark.asyncio
async def test_v0_vertical_slice_endtoend():
    pytest.importorskip("kaos_pdf")
    pytest.importorskip("kaos_nlp_transformers")
    pytest.importorskip("kaos_llm_core")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        pytest.skip("ANTHROPIC_API_KEY required for live LLM call")

    from kaos_content.views import DocumentView
    from kaos_pdf import extract_pdf  # type: ignore[import-not-found]

    from kaos_ml_core import Corpus
    from kaos_ml_core.cluster import kmedoid_seeds, minibatch_kmeans
    from kaos_ml_core.features import embed_corpus
    from kaos_ml_core.label import label_seeds_with_llm
    from kaos_ml_core.predict import predict_corpus
    from kaos_ml_core.train import train_logreg

    # 1. Load real PDFs from the kaos-pdf fixture set. The fixture
    #    directory deliberately contains a corrupt file (bad_test1.pdf)
    #    used by kaos-pdf's own error-handling tests; skip files that
    #    fail to extract so we don't trip on them.
    pdf_paths = _kaos_pdf_fixtures()
    if not pdf_paths:
        pytest.skip("no kaos-pdf fixtures on disk")

    docs = []
    for p in pdf_paths:
        try:
            docs.append(extract_pdf(p))
        except Exception:
            continue
        if len(docs) >= 5:
            break

    docs_by_uri = {d.metadata.source.uri: d for d in docs if d.metadata.source is not None}
    assert docs_by_uri, "expected at least one extracted document with a source URI"

    # 2. Build the Corpus and verify the AST round-trip on a random sample
    corpus = Corpus.from_documents(docs, level="paragraph")
    assert len(corpus) >= 30, f"expected >=30 paragraph units, got {len(corpus)}"

    rng = random.Random(42)
    sample = rng.sample(range(len(corpus)), min(20, len(corpus)))
    for row in sample:
        unit = corpus.unit(row)
        assert corpus.block_ref_for(row) == unit.block_ref
        assert corpus.row_for(unit.block_ref) <= row
        # The block_ref resolves back through DocumentView in the original doc
        view = DocumentView(docs_by_uri[unit.doc_uri])
        assert any(p.block_ref == unit.block_ref for p in view.paragraphs)

    # 3. Embed via fastembed
    X = embed_corpus(corpus, model="BAAI/bge-small-en-v1.5")
    assert X.shape == (len(corpus), 384)

    # 4. Cluster + cold-start seed selection
    clusters = minibatch_kmeans(X, n_clusters=min(10, max(2, len(corpus) // 5)))
    seeds = kmedoid_seeds(X, clusters, per_cluster=3)
    assert 1 <= len(seeds) <= len(corpus)

    # 5. Live LLM labels (Haiku — cheapest current-gen Anthropic model)
    labels = await label_seeds_with_llm(
        corpus,
        seeds,
        classes=["legal", "non_legal"],
        instructions=(
            "Classify whether the paragraph is from a legal document "
            "(contract, statute, court filing, regulation) or general prose."
        ),
        model="claude-haiku-4-5",
    )
    assert len(set(labels.values())) >= 2, (
        "need at least 2 distinct classes labeled to train a binary classifier"
    )

    # 6. Train + apply
    clf = train_logreg(X, labels)
    predictions = predict_corpus(corpus, X, clf, threshold=0.5)

    # 7. The whole thing round-trips back to AST addresses
    assert len(predictions.tables) == 1
    pred_table = predictions.tables[0]
    assert pred_table.name == "predictions"
    assert len(pred_table.rows) == len(corpus)

    # Verify a random sample of predictions resolves back to real paragraphs
    pred_sample = rng.sample(list(pred_table.rows), min(20, len(pred_table.rows)))
    for r in pred_sample:
        # Row tuple shape:
        #   (row, block_ref, doc_uri, page, section_ref,
        #    section_title, predicted_label, score, above_threshold)
        block_ref = r[1]
        doc_uri = r[2]
        doc = docs_by_uri[doc_uri]
        view = DocumentView(doc)
        assert any(p.block_ref == block_ref for p in view.paragraphs), (
            f"prediction block_ref {block_ref} not found in doc {doc_uri}"
        )


def _kaos_pdf_fixtures() -> list:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "kaos-pdf" / "tests" / "fixtures"
    if not root.is_dir():
        return []
    return sorted(root.glob("*.pdf"))
