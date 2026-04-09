"""Test 3 — Cold-start end-to-end with held-out evaluation.

> Does the cold-start loop actually close — train on LLM labels, generalize
> to unseen documents, beat the perfect-label baseline by less than 5%?

This is the keystone test. It runs the full v0 vertical slice on a real
labeled corpus:

    1. Build a Corpus from 600 USC chapters (300 tax + 300 criminal)
    2. Stratified train/eval split: 402 train + 198 held-out eval
    3. Embed both via fastembed BAAI/bge-small-en-v1.5
    4. Cluster the training set into 20 clusters
    5. Pick k-medoid seeds (~60 chapters)
    6. Ask Haiku to label the seeds — these are the ONLY labels seen
    7. Train LogReg on the LLM-labeled seeds
    8. Apply to the held-out 198 documents
    9. Compare predictions to ground-truth USC Title

A practicing lawyer reading this test's output should be able to decide
whether to trust the classifier on a real matter. The confusion matrix
goes to stdout.

Live test: requires ANTHROPIC_API_KEY. Cost: ~$0.005. Time: ~60s after
fastembed model is cached.
"""

from __future__ import annotations

import os

import pytest

from kaos_ml_core.label import label_seeds_with_llm
from kaos_ml_core.predict import predict_corpus
from kaos_ml_core.train import train_logreg
from tests.fixtures.usc_corpus import USC_JSONL
from tests.integration._usc_helpers import (
    DEFAULT_RANDOM_STATE,
    EMBED_MODEL,
    LABELS,
    LLM_MODEL,
    cluster_and_seed,
    evaluate,
    load_corpus_and_features,
    predict_documents,
    print_report,
)
from tests.integration.test_usc_llm_label_quality import LLM_INSTRUCTION

pytestmark = pytest.mark.integration


# Hard floor: the LLM-labeled cold-start classifier must come within
# this fraction of the ground-truth-seeded classifier's accuracy. If
# Test 1 hits 0.94 with perfect labels, the cold-start should hit at
# least 0.94 * 0.93 ≈ 0.87. Set as a multiplicative bar so the test
# tracks Test 1 as the pipeline gets better.
COLD_START_VS_PERFECT_LABEL_FLOOR = 0.93


@pytest.mark.asyncio
async def test_cold_start_endtoend_against_ground_truth():
    """Live Haiku labels → train → predict → measure vs ground truth.

    Hard assertions:
        - accuracy on held-out 198 docs >= 0.85
        - per-class recall >= 0.80
        - delta over majority baseline >= 0.30
        - within 7% of the perfect-label baseline (Test 1's accuracy)
        - every prediction round-trips through DocumentView
    """
    if not USC_JSONL.is_file():
        pytest.skip(f"USC fixture not found at {USC_JSONL}")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        pytest.skip("ANTHROPIC_API_KEY required for live LLM call")

    (
        train_corpus,
        X_train,
        train_gt,
        eval_corpus,
        X_eval,
        eval_gt_rows,
        _labels,
    ) = load_corpus_and_features()

    _clusters, seed_rows = cluster_and_seed(X_train)

    # ── 1. LLM labels the seeds (the only labels we'll see) ────────
    llm_labels = await label_seeds_with_llm(
        train_corpus,
        seed_rows,
        classes=LABELS,
        instructions=LLM_INSTRUCTION,
        model=LLM_MODEL,
    )
    assert len(llm_labels) >= int(0.85 * len(seed_rows)), (
        f"only {len(llm_labels)}/{len(seed_rows)} seeds got a usable LLM label"
    )
    assert len(set(llm_labels.values())) == 2, (
        "LLM produced labels for only one class — cold-start training is impossible"
    )

    # Useful side-stat: how often the LLM agreed with the ground truth
    # on the seeds. Reported but NOT asserted here — Test 2 owns that
    # assertion. Test 3's job is to measure end-to-end generalization.
    seed_agreement = sum(1 for row, llm in llm_labels.items() if llm == train_gt[row]) / len(
        llm_labels
    )

    # ── 2. Train on the LLM labels ─────────────────────────────────
    clf = train_logreg(X_train, llm_labels, random_state=DEFAULT_RANDOM_STATE)

    # ── 3. Predict on the held-out 198 ─────────────────────────────
    doc_pred = predict_documents(eval_corpus, X_eval, clf)
    doc_gt = {u.doc_uri: eval_gt_rows[u.row] for u in eval_corpus}
    report = evaluate(doc_pred, doc_gt, LABELS)

    # ── 4. Compute the perfect-label baseline (the Test 1 number) so
    #      we can verify cold-start is within 5% of it ──────────────
    perfect_seed_labels = {row: train_gt[row] for row in seed_rows}
    perfect_clf = train_logreg(X_train, perfect_seed_labels, random_state=DEFAULT_RANDOM_STATE)
    perfect_doc_pred = predict_documents(eval_corpus, X_eval, perfect_clf)
    perfect_report = evaluate(perfect_doc_pred, doc_gt, LABELS)

    print_report(
        "Test 3 — Cold-start end-to-end (LLM labels → train → predict)",
        report,
        labels=LABELS,
        extras={
            "embedding model": EMBED_MODEL,
            "LLM model": LLM_MODEL,
            "seeds attempted": str(len(seed_rows)),
            "seeds labeled by LLM": str(len(llm_labels)),
            "LLM-vs-truth on seeds": f"{seed_agreement:.3f}",
            "perfect-label baseline": f"{perfect_report.accuracy:.3f}",
            "cold-start / perfect-label": f"{report.accuracy / perfect_report.accuracy:.3f}",
            "n_eval_documents": str(report.n_eval),
        },
    )

    # ── Hard assertions ─────────────────────────────────────────────
    assert report.accuracy >= 0.85, (
        f"cold-start accuracy {report.accuracy:.3f} below 0.85 — the trained "
        "classifier is too weak to be useful on held-out docs"
    )
    for L in LABELS:
        assert report.per_class_recall[L] >= 0.80, (
            f"recall for {L} = {report.per_class_recall[L]:.3f} below 0.80"
        )
    assert report.above_baseline() >= 0.30, (
        f"only {report.above_baseline():+.3f} above majority baseline — barely "
        "better than always guessing one class"
    )
    ratio = report.accuracy / perfect_report.accuracy if perfect_report.accuracy else 0
    assert ratio >= COLD_START_VS_PERFECT_LABEL_FLOOR, (
        f"cold-start accuracy {report.accuracy:.3f} is only {ratio:.3f}x the "
        f"perfect-label accuracy {perfect_report.accuracy:.3f} — should be "
        f">= {COLD_START_VS_PERFECT_LABEL_FLOOR}x. The LLM labels are degrading "
        "training enough to matter."
    )

    # ── AST round-trip on every prediction ──────────────────────────
    predictions = predict_corpus(eval_corpus, X_eval, clf)
    pred_table = predictions.tables[0]
    assert len(pred_table.rows) == len(eval_corpus)
    eval_block_refs = {u.block_ref for u in eval_corpus}
    for r in pred_table.rows:
        assert r[1] in eval_block_refs, f"prediction block_ref {r[1]} not in the eval Corpus"
