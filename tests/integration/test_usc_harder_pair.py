"""Test 4 — Harder title pair (Title 26 tax vs Title 12 banking).

> Does the pipeline still work when the two classes share vocabulary?

The default acceptance gate uses Title 26 (tax) vs Title 18 (criminal),
which is essentially trivial — the two domains share almost no
vocabulary, so even a TF-IDF baseline would do well. This test runs the
same pipeline against Title 26 (tax) vs Title 12 (banking), which both
involve money, financial institutions, IRS/Treasury vocabulary, and
regulatory procedure.

A pipeline that hits > 0.85 here is genuinely useful, not just
exploiting an obvious domain gap. Looser thresholds than the easy pair:
0.85 accuracy floor (vs 0.90), 0.75 per-class recall floor (vs 0.85).

Live LLM call. Cost: ~$0.005. Time: ~60s additional after the easy-pair
embedding cache is warm.
"""

from __future__ import annotations

import os

import pytest

from kaos_ml_core.label import label_seeds_with_llm
from kaos_ml_core.train import train_logreg
from tests.fixtures.usc_corpus import (
    LABEL_BANKS,
    LABEL_TAX,
    TITLE_BANKS,
    TITLE_TAX,
    USC_JSONL,
)
from tests.integration._usc_helpers import (
    DEFAULT_RANDOM_STATE,
    EMBED_MODEL,
    LLM_MODEL,
    cluster_and_seed,
    evaluate,
    load_corpus_and_features,
    predict_documents,
    print_report,
)

pytestmark = pytest.mark.integration


HARDER_TITLES: tuple[int, ...] = (TITLE_BANKS, TITLE_TAX)
HARDER_LABELS: list[str] = [LABEL_BANKS, LABEL_TAX]

HARDER_INSTRUCTION = (
    f"Decide whether the following passage is from United States Code "
    f"{LABEL_TAX} (federal tax law — Internal Revenue Code, income tax, "
    "deductions, IRS procedure, exempt organizations, excise taxes) or "
    f"{LABEL_BANKS} (federal banking law — chartering and supervision of "
    "national banks, Federal Reserve, FDIC insurance, holding companies, "
    "credit unions, banking transactions). "
    f"Reply with exactly one of the two labels: '{LABEL_TAX}' or "
    f"'{LABEL_BANKS}'. Both topics involve money, but the focus is "
    "different: tax law is about raising revenue, banking law is about "
    "regulating financial institutions."
)


@pytest.mark.asyncio
async def test_harder_pair_cold_start_endtoend():
    """Run the full cold-start pipeline on tax vs banking.

    Hard assertions:
        - cold-start accuracy >= 0.85
        - per-class recall >= 0.75
        - >= 0.30 above majority baseline
        - LLM-vs-truth seed agreement >= 0.80 (Haiku is competent here too)
        - cold-start within 7% of perfect-label baseline
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
    ) = load_corpus_and_features(titles=HARDER_TITLES)

    _clusters, seed_rows = cluster_and_seed(X_train)

    # ── 1. LLM labels ───────────────────────────────────────────────
    llm_labels = await label_seeds_with_llm(
        train_corpus,
        seed_rows,
        classes=HARDER_LABELS,
        instructions=HARDER_INSTRUCTION,
        model=LLM_MODEL,
    )
    assert len(llm_labels) >= int(0.85 * len(seed_rows)), (
        f"only {len(llm_labels)}/{len(seed_rows)} seeds got a usable LLM label"
    )

    seed_agreement = sum(1 for row, llm in llm_labels.items() if llm == train_gt[row]) / max(
        len(llm_labels), 1
    )

    if len(set(llm_labels.values())) < 2:
        pytest.fail(
            f"LLM labeled all seeds as one class on the harder pair "
            f"(distribution: {dict.fromkeys(set(llm_labels.values()), '...')})"
        )

    # ── 2. Train on LLM labels ──────────────────────────────────────
    clf = train_logreg(X_train, llm_labels, random_state=DEFAULT_RANDOM_STATE)

    # ── 3. Predict + evaluate at document level ────────────────────
    doc_pred = predict_documents(eval_corpus, X_eval, clf)
    doc_gt = {u.doc_uri: eval_gt_rows[u.row] for u in eval_corpus}
    report = evaluate(doc_pred, doc_gt, HARDER_LABELS)

    # ── 4. Perfect-label baseline so we can measure cold-start delta ─
    perfect_seed_labels = {row: train_gt[row] for row in seed_rows}
    perfect_clf = train_logreg(X_train, perfect_seed_labels, random_state=DEFAULT_RANDOM_STATE)
    perfect_doc_pred = predict_documents(eval_corpus, X_eval, perfect_clf)
    perfect_report = evaluate(perfect_doc_pred, doc_gt, HARDER_LABELS)

    print_report(
        "Test 4 — Harder pair: tax (Title 26) vs banking (Title 12)",
        report,
        labels=HARDER_LABELS,
        extras={
            "embedding model": EMBED_MODEL,
            "LLM model": LLM_MODEL,
            "title pair": f"{TITLE_TAX}-tax vs {TITLE_BANKS}-banks",
            "seeds attempted": str(len(seed_rows)),
            "seeds labeled by LLM": str(len(llm_labels)),
            "LLM-vs-truth on seeds": f"{seed_agreement:.3f}",
            "perfect-label baseline": f"{perfect_report.accuracy:.3f}",
            "cold-start / perfect-label": f"{report.accuracy / perfect_report.accuracy:.3f}",
            "n_eval_documents": str(report.n_eval),
        },
    )

    # ── Hard assertions (looser than the easy pair) ────────────────
    assert report.accuracy >= 0.85, (
        f"harder-pair cold-start accuracy {report.accuracy:.3f} below 0.85 — "
        "the pipeline cannot distinguish tax from banking with LLM-only labels"
    )
    for L in HARDER_LABELS:
        assert report.per_class_recall[L] >= 0.75, (
            f"recall for {L} = {report.per_class_recall[L]:.3f} below 0.75"
        )
    assert report.above_baseline() >= 0.30, (
        f"only {report.above_baseline():+.3f} above majority baseline — "
        "barely better than always guessing one class"
    )
    assert seed_agreement >= 0.80, (
        f"Haiku-vs-truth seed agreement {seed_agreement:.3f} below 0.80 — "
        "the LLM is confused by the harder vocabulary overlap"
    )
    ratio = report.accuracy / perfect_report.accuracy if perfect_report.accuracy else 0
    assert ratio >= 0.93, (
        f"cold-start accuracy {report.accuracy:.3f} is only {ratio:.3f}x the "
        f"perfect-label accuracy {perfect_report.accuracy:.3f} — should be >= 0.93x"
    )
