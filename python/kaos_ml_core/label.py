"""LLM-driven cold-start labeling for a Corpus.

In v0 this calls ``kaos_llm_core.starter.classify`` directly per seed —
the simplest possible thing. v1.3 wires it to
``kaos_llm_core.batch_run()`` for the crash-safe resume + Budget pattern
(mirroring the kaos-llm-core Phase 15.2 contract).
"""

from __future__ import annotations

import importlib

from kaos_core.logging import get_logger

from kaos_ml_core.corpus import Corpus
from kaos_ml_core.errors import LabelError

logger = get_logger(__name__)


async def label_seeds_with_llm(
    corpus: Corpus,
    seed_rows: list[int],
    *,
    classes: list[str],
    instructions: str,
    model: str = "claude-haiku-4-5",
) -> dict[int, str]:
    """Label a small set of corpus rows via the kaos-llm-core starter API.

    Args:
        corpus: The Corpus the rows refer to.
        seed_rows: Row indices to label (typically from
            ``kmedoid_seeds()``).
        classes: The set of allowed labels (must be at least 2).
        instructions: Task instructions passed to ``starter.classify``.
        model: LLM model id (default: ``claude-haiku-4-5`` —
            cheapest current-gen Anthropic model).

    Returns:
        A dict mapping row index to assigned label string. Rows where
        the LLM returned an out-of-set label or where the call failed
        are dropped with a ``logger.warning()`` identifying the row
        and reason — those rows will not appear in the returned dict
        and will not be used for training. v1.3 makes failures
        structured via the crash-safe JSONL log.

    Raises:
        LabelError: If the ``[llm]`` extra is not installed, the
            classes list has fewer than 2 entries, or zero rows
            received a usable label.
    """
    try:
        starter = importlib.import_module("kaos_llm_core.starter")
    except ImportError as exc:
        msg = (
            "label_seeds_with_llm requires the [llm] extra. "
            "Fix: install kaos-ml-core[llm]. "
            "Alternative: provide labels directly to train_logreg() if you "
            "already have them."
        )
        raise LabelError(msg) from exc

    classify = starter.classify
    if not classes or len(classes) < 2:
        msg = (
            f"classify requires at least 2 classes; got {len(classes)}. "
            "Fix: provide a list of class labels, e.g. ['responsive', 'non_responsive']."
        )
        raise LabelError(msg)

    classes_set = set(classes)
    labels: dict[int, str] = {}

    for row in seed_rows:
        unit = corpus.unit(row)
        try:
            result = await classify(
                text=unit.text,
                labels=classes,
                instruction=instructions,
                model=model,
            )
        except Exception:
            # Soft failure — skip this row, continue. v1.3 makes this
            # structured via the JSONL log.
            logger.warning(
                "LLM labeling failed for row %d (block_ref=%s): %s. Row dropped from seed labels.",
                row,
                unit.block_ref,
                "classify call raised an exception",
            )
            continue

        # starter.classify returns str (single-label) or list[str] (multi_label).
        # We always call single-label here, so result is a str.
        if isinstance(result, str) and result in classes_set:
            labels[row] = result
        else:
            logger.warning(
                "LLM returned out-of-set label for row %d (block_ref=%s): "
                "got %r, expected one of %s. Row dropped from seed labels.",
                row,
                unit.block_ref,
                result,
                sorted(classes_set),
            )

    if not labels:
        msg = (
            "LLM labeling produced zero usable labels. "
            "Fix: check that the LLM provider credentials are configured "
            "(KAOS_LLM_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY for Haiku) "
            "and that the model returned values match the classes list. "
            "Alternative: try a smaller per_cluster value in kmedoid_seeds() "
            "or different cluster instructions."
        )
        raise LabelError(msg)

    return labels


__all__ = ["label_seeds_with_llm"]
