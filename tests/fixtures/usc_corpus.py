"""USC fixture loader for the kaos-ml-core test suite.

Reads ``kaos-nlp-core/tests/fixtures/usc.jsonl`` (68k+ chapter-level
records of the United States Code, sourced from data.kl3m.ai), filters
to a balanced binary subset of two well-separated titles, and wraps each
record as a ``ContentDocument`` with a stable ``doc_uri`` so the
ground-truth title label survives the round-trip through ``Corpus``.

This is the dataset that powers the kaos-ml-core acceptance gate. The
ground truth is unambiguous (every chapter belongs to exactly one Title)
and the labels are NOT printed verbatim in the text — the Title number
appears nowhere in the chapter body, only in the s3 path metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from kaos_content.model.attr import SourceRef
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata

# Two well-separated titles for the easy pair. Tax law and criminal law
# share almost no vocabulary — a working pipeline should hit > 0.90 on
# this binary.
TITLE_TAX = 26  # Internal Revenue Code
TITLE_CRIMINAL = 18  # Crimes and Criminal Procedure

# Harder pair: tax vs banking. Both deal with money, both regulate
# financial institutions, both have IRS/Treasury vocabulary. A pipeline
# that hits > 0.85 here is genuinely useful, not just exploiting an easy
# domain gap.
TITLE_BANKS = 12  # Banks and Banking

# Stable label strings used by tests, the LLM, and the trained classifier.
LABEL_TAX = "tax_law"
LABEL_CRIMINAL = "criminal_law"
LABEL_BANKS = "banking_law"

TITLE_TO_LABEL: dict[int, str] = {
    TITLE_TAX: LABEL_TAX,
    TITLE_CRIMINAL: LABEL_CRIMINAL,
    TITLE_BANKS: LABEL_BANKS,
}

# Path to the kaos-nlp-core fixture file. The kaos-ml-core test suite
# depends on this being downloaded (see kaos-nlp-core/tests/fixtures/
# download_hf_fixtures.py).
USC_JSONL = (
    Path(__file__).resolve().parents[3] / "kaos-nlp-core" / "tests" / "fixtures" / "usc.jsonl"
)

# Identifier format from data.kl3m.ai:
#   s3://data.kl3m.ai/documents/usc/<congress>/<session>/<title>_-...
_TITLE_RE = re.compile(r"/usc/\d+/\d+/(\d+)[_-]")


@dataclass(frozen=True, slots=True)
class USCRecord:
    """One USC chapter record with parsed title and ground-truth label."""

    record_id: int
    identifier: str  # the s3:// path
    title: int  # Title number (e.g. 26 for tax)
    label: str  # ground-truth label string
    text: str  # chapter body text


def load_usc_records(
    *,
    titles: tuple[int, ...] = (TITLE_TAX, TITLE_CRIMINAL),
    per_title: int = 300,
    seed: int = 42,
) -> list[USCRecord]:
    """Load a balanced subset of USC chapter records.

    Deterministic for a given (titles, per_title, seed). Skips empty
    or trivial records (< 200 chars) so we don't poison clustering
    with stub chapters.
    """
    if not USC_JSONL.is_file():
        msg = (
            f"USC fixture not found at {USC_JSONL}. "
            "Fix: run `python kaos-nlp-core/tests/fixtures/download_hf_fixtures.py` "
            "from the repo root, which downloads usc.jsonl, edgar_agreements.jsonl, "
            "and patents.jsonl into the kaos-nlp-core fixtures directory."
        )
        raise FileNotFoundError(msg)

    by_title: dict[int, list[USCRecord]] = {t: [] for t in titles}

    with USC_JSONL.open() as fh:
        for line in fh:
            d = json.loads(line)
            m = _TITLE_RE.search(d.get("identifier", ""))
            if not m:
                continue
            t = int(m.group(1))
            if t not in by_title:
                continue
            text = d.get("text", "")
            if not text or len(text) < 200:
                continue
            by_title[t].append(
                USCRecord(
                    record_id=int(d["id"]),
                    identifier=d["identifier"],
                    title=t,
                    label=TITLE_TO_LABEL[t],
                    text=text,
                )
            )

    # Deterministic subsample to per_title records each
    import random

    rng = random.Random(seed)
    selected: list[USCRecord] = []
    for t in titles:
        pool = by_title[t]
        if len(pool) < per_title:
            msg = (
                f"USC fixture has only {len(pool)} usable records for Title {t}, "
                f"need {per_title}. "
                "Fix: lower per_title in the test, or pick a different title pair."
            )
            raise ValueError(msg)
        selected.extend(rng.sample(pool, per_title))

    rng.shuffle(selected)
    return selected


def usc_record_to_document(
    rec: USCRecord,
    *,
    max_chars: int = 8000,
) -> ContentDocument:
    """Wrap one USC chapter record as a ContentDocument.

    Each chapter becomes a single ``Paragraph`` containing the full
    chapter text (truncated to ``max_chars`` to keep embedding cost
    bounded — bge-small's 512-token window only sees the first ~1500
    chars anyway). This is the right granularity for the v0 acceptance
    gate: one feature row per chapter, one ground-truth label per
    chapter, one ``block_ref`` per chapter that round-trips through
    ``DocumentView``.

    The multi-paragraph split is exercised separately by
    ``tests/unit/test_corpus.py``, where it belongs.

    The ``doc_uri`` is the original s3 path, which is what the LLM and
    the classifier never see — the Title number lives in the URI, not
    in the body. That's how we keep the ground truth honest.
    """
    text = rec.text.strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    return ContentDocument(
        metadata=DocumentMetadata(source=SourceRef(uri=rec.identifier)),
        body=(Paragraph(children=(Text(value=text),)),),
    )


def load_usc_corpus(
    *,
    titles: tuple[int, ...] = (TITLE_TAX, TITLE_CRIMINAL),
    per_title: int = 300,
    seed: int = 42,
) -> tuple[list[ContentDocument], dict[str, str]]:
    """Load USC records and build (documents, doc_uri → ground_truth label).

    Args:
        titles: USC Title numbers to include. Default is the easy pair
            (26 tax + 18 criminal). For a harder benchmark try
            ``(TITLE_TAX, TITLE_BANKS)`` — both financial-regulation
            domains.
        per_title: Number of records per title. Must not exceed the
            number of usable records in the smallest selected title.
        seed: Deterministic subsample seed.

    Returns:
        documents: List of ContentDocument, deterministically ordered.
        ground_truth: dict[doc_uri -> label_string] mapping each
            document's source URI to its ground-truth label.
    """
    records = load_usc_records(titles=titles, per_title=per_title, seed=seed)
    docs = [usc_record_to_document(r) for r in records]
    ground_truth = {r.identifier: r.label for r in records}
    return docs, ground_truth


def split_train_eval(
    docs: list[ContentDocument],
    ground_truth: dict[str, str],
    *,
    eval_fraction: float = 0.33,
    seed: int = 42,
) -> tuple[list[ContentDocument], list[ContentDocument]]:
    """Deterministic train/eval split, stratified by label.

    Returns (train_docs, eval_docs). The split is stratified so each
    label is represented proportionally in both halves — required for
    the evaluation to be meaningful.
    """
    import random

    rng = random.Random(seed)

    by_label: dict[str, list[ContentDocument]] = {}
    for d in docs:
        src = d.metadata.source
        if src is None:
            msg = "split_train_eval requires every document to carry metadata.source"
            raise ValueError(msg)
        label = ground_truth[src.uri]
        by_label.setdefault(label, []).append(d)

    train: list[ContentDocument] = []
    eval_: list[ContentDocument] = []
    for label_docs in by_label.values():
        shuffled = list(label_docs)
        rng.shuffle(shuffled)
        n_eval = round(len(shuffled) * eval_fraction)
        eval_.extend(shuffled[:n_eval])
        train.extend(shuffled[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


__all__ = [
    "LABEL_BANKS",
    "LABEL_CRIMINAL",
    "LABEL_TAX",
    "TITLE_BANKS",
    "TITLE_CRIMINAL",
    "TITLE_TAX",
    "TITLE_TO_LABEL",
    "USCRecord",
    "load_usc_corpus",
    "load_usc_records",
    "split_train_eval",
    "usc_record_to_document",
]
