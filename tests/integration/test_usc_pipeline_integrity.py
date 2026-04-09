"""Test 1 — Pipeline integrity with ground-truth seeds.

> If the labels are perfect, does the pipeline produce a useful classifier?

No LLM. Deterministic. Uses ground-truth USC Title labels for the 50-60
k-medoid seeds and trains on them, then evaluates on the held-out third
of the corpus. This is the test that catches structural bugs in the
pipeline (featurization, training, AST row alignment) — if it passes,
the pipeline is structurally sound; if it fails, every other test in
this file is uninterpretable.

Marked as ``integration`` because it embeds 600 documents via fastembed
on first run (~30s once the ONNX model is cached). The LLM-free nature
means it runs without API credentials and is fully deterministic.
"""

from __future__ import annotations

import pytest

from kaos_ml_core.predict import predict_corpus
from kaos_ml_core.train import train_logreg
from tests.fixtures.usc_corpus import USC_JSONL
from tests.integration._usc_helpers import (
    DEFAULT_RANDOM_STATE,
    LABELS,
    SEEDS_PER_CLUSTER,
    cluster_and_seed,
    evaluate,
    load_corpus_and_features,
    predict_documents,
    print_report,
)

pytestmark = pytest.mark.integration


def test_pipeline_integrity_with_ground_truth_seeds():
    """Train on perfect labels for k-medoid seeds; assert it generalizes.

    Hard assertions:
        - accuracy on held-out 198 documents >= 0.90
        - per-class recall >= 0.85
        - delta over majority baseline >= 0.30
        - every prediction round-trips through DocumentView
    """
    if not USC_JSONL.is_file():
        pytest.skip(f"USC fixture not found at {USC_JSONL}")

    (
        train_corpus,
        X_train,
        train_gt,
        eval_corpus,
        X_eval,
        eval_gt_rows,
        _labels,
    ) = load_corpus_and_features()

    # Pick k-medoid seeds and look up GROUND TRUTH labels (no LLM)
    _clusters, seed_rows = cluster_and_seed(X_train)
    seed_labels = {row: train_gt[row] for row in seed_rows}

    # Sanity: at least both classes in the seed set
    assert len(set(seed_labels.values())) == 2, (
        f"k-medoid seeds didn't cover both classes: {seed_labels}"
    )

    clf = train_logreg(X_train, seed_labels, random_state=DEFAULT_RANDOM_STATE)

    # ── Document-level evaluation on held-out ───────────────────────
    doc_pred = predict_documents(eval_corpus, X_eval, clf)
    doc_gt = {u.doc_uri: eval_gt_rows[u.row] for u in eval_corpus}
    report = evaluate(doc_pred, doc_gt, LABELS)

    print_report(
        "Test 1 — Pipeline integrity (ground-truth seeds, no LLM)",
        report,
        labels=LABELS,
        extras={
            "n_seeds": str(len(seed_labels)),
            "n_train_paragraph_rows": str(len(train_corpus)),
            "n_eval_documents": str(len(doc_gt)),
            "n_eval_paragraph_rows": str(len(eval_corpus)),
            "embedding model": "BAAI/bge-small-en-v1.5",
            "classifier": "LogReg(liblinear, balanced)",
            "seeds_per_cluster": str(SEEDS_PER_CLUSTER),
        },
    )

    # ── Hard assertions ─────────────────────────────────────────────
    assert report.accuracy >= 0.90, (
        f"accuracy={report.accuracy:.3f} below the 0.90 bar with perfect labels — "
        "this means the pipeline structure is broken (featurization, training, "
        "or AST row alignment)"
    )
    for L in LABELS:
        assert report.per_class_recall[L] >= 0.85, (
            f"recall for {L} = {report.per_class_recall[L]:.3f} below 0.85"
        )
    assert report.above_baseline() >= 0.30, (
        f"only {report.above_baseline():+.3f} above majority baseline "
        f"({report.majority_baseline:.3f}) — barely better than always guessing one class"
    )

    # ── AST round-trip on every prediction ──────────────────────────
    predictions = predict_corpus(eval_corpus, X_eval, clf)
    pred_table = predictions.tables[0]
    assert len(pred_table.rows) == len(eval_corpus)
    eval_block_refs = {u.block_ref for u in eval_corpus}
    for r in pred_table.rows:
        # row tuple: (row, block_ref, doc_uri, page, section_ref,
        #             section_title, predicted_label, score, above_threshold)
        assert r[1] in eval_block_refs, f"prediction block_ref {r[1]} not in the eval Corpus"
