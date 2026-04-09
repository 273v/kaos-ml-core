"""Test 2 — LLM label quality vs ground truth.

> Is Haiku competent at this labeling task?

Asks claude-haiku-4-5 to label the same 50-60 k-medoid seed chapters
that Test 1 trained on, then measures how often the LLM's label matches
the ground-truth USC Title. This isolates the cold-start labeling step
from the training step — if Haiku scores 0.60 here, the prompt is the
bottleneck and we'd be training on noise downstream.

Live test: requires ANTHROPIC_API_KEY (or KAOS_LLM_ANTHROPIC_API_KEY).
Auto-skips when credentials are missing. Cost: ~$0.005.
"""

from __future__ import annotations

import os

import pytest

from kaos_ml_core.label import label_seeds_with_llm
from tests.fixtures.usc_corpus import (
    LABEL_CRIMINAL,
    LABEL_TAX,
    USC_JSONL,
)
from tests.integration._usc_helpers import (
    LABELS,
    LLM_MODEL,
    cluster_and_seed,
    load_corpus_and_features,
)

pytestmark = pytest.mark.integration


# Single-paragraph corpora produced by usc_record_to_document mean each
# row is exactly one chapter, so we can read the full chapter text from
# corpus.unit(row).text without further joining. Instructions to the
# LLM avoid leaking the title number even by example.
LLM_INSTRUCTION = (
    "Decide whether the following passage is from United States Code "
    f"{LABEL_TAX} (federal tax law — Internal Revenue Code, income tax, "
    "deductions, IRS procedure, exempt organizations, excise taxes) or "
    f"{LABEL_CRIMINAL} (federal criminal law — crimes, criminal procedure, "
    "prosecutions, prison, fines as punishment for crimes). "
    f"Reply with exactly one of the two labels: '{LABEL_TAX}' or '{LABEL_CRIMINAL}'."
)


@pytest.mark.asyncio
async def test_llm_label_quality_against_ground_truth():
    """Live Haiku call on the k-medoid seeds; assert agreement with truth.

    Hard assertions:
        - >= 90% of seed rows received a usable label (>= 0.90 completion)
        - >= 85% of usable labels match ground truth (>= 0.85 agreement)
        - both classes are represented in the LLM's labels
    """
    if not USC_JSONL.is_file():
        pytest.skip(f"USC fixture not found at {USC_JSONL}")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("KAOS_LLM_ANTHROPIC_API_KEY")):
        pytest.skip("ANTHROPIC_API_KEY required for live LLM call")

    (
        train_corpus,
        X_train,
        train_gt,
        _eval_corpus,
        _X_eval,
        _eval_gt,
        _labels,
    ) = load_corpus_and_features()
    _clusters, seed_rows = cluster_and_seed(X_train)

    # Live LLM call
    llm_labels = await label_seeds_with_llm(
        train_corpus,
        seed_rows,
        classes=LABELS,
        instructions=LLM_INSTRUCTION,
        model=LLM_MODEL,
    )

    n_seeds = len(seed_rows)
    n_labeled = len(llm_labels)
    completion_rate = n_labeled / n_seeds if n_seeds else 0.0

    # Agreement on the SUBSET that the LLM successfully labeled
    agreements = sum(1 for row, llm_label in llm_labels.items() if llm_label == train_gt[row])
    agreement_rate = agreements / n_labeled if n_labeled else 0.0

    # Disagreements (so the report shows what the LLM got wrong)
    disagreements = [
        (row, train_gt[row], llm_labels[row])
        for row in llm_labels
        if llm_labels[row] != train_gt[row]
    ]

    # Class coverage in the LLM's labels
    label_distribution = dict.fromkeys(LABELS, 0)
    for L in llm_labels.values():
        label_distribution[L] = label_distribution.get(L, 0) + 1

    # ── Pretty-print the report ─────────────────────────────────────
    print("\nTest 2 — LLM label quality vs ground truth")
    print("==========================================")
    print(f"  model:                        {LLM_MODEL}")
    print(f"  seeds attempted:              {n_seeds}")
    print(f"  seeds labeled:                {n_labeled}")
    print(f"  completion rate:              {completion_rate:.3f}")
    print(f"  agreement w/ ground truth:    {agreements}/{n_labeled} = {agreement_rate:.3f}")
    print()
    print("  LLM label distribution:")
    for L in LABELS:
        print(f"    {L:<20s} {label_distribution[L]}")
    print()
    if disagreements:
        print(f"  disagreements ({len(disagreements)}):")
        for row, true_label, llm_label in disagreements[:10]:
            doc_uri = train_corpus.unit(row).doc_uri
            snippet = train_corpus.unit(row).text[:120].replace("\n", " ")
            print(f"    row={row} truth={true_label} llm={llm_label}")
            print(f"      doc_uri={doc_uri}")
            print(f"      text={snippet}...")
        if len(disagreements) > 10:
            print(f"    ... and {len(disagreements) - 10} more")
    print()

    # ── Hard assertions ─────────────────────────────────────────────
    assert completion_rate >= 0.90, (
        f"completion={completion_rate:.3f} below 0.90 — Haiku is failing on "
        "too many seeds. Check the LLM provider credentials and the prompt."
    )
    assert agreement_rate >= 0.85, (
        f"LLM-vs-ground-truth agreement {agreement_rate:.3f} below 0.85 — "
        "Haiku is not competent at this task with the current prompt; "
        "the cold-start training step would be training on noise."
    )
    assert label_distribution[LABEL_TAX] > 0, (
        "LLM never produced the tax_law label — class collapse"
    )
    assert label_distribution[LABEL_CRIMINAL] > 0, (
        "LLM never produced the criminal_law label — class collapse"
    )
