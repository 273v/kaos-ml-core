"""Feature matrix construction for a Corpus.

In v0 we have one path: dense embeddings via the sibling
``kaos-nlp-transformers`` package (which wraps fastembed). v1.1 adds a
sparse TF-IDF path that consumes ``kaos-nlp-core``'s existing
``InvertedIndex`` and ``SparseTermMatrix`` (already implemented in Rust
on that side — no new Rust hot path needed in this package).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from kaos_ml_core.corpus import Corpus
from kaos_ml_core.errors import FeatureError
from kaos_ml_core.settings import KaosMLCoreSettings


def embed_corpus(
    corpus: Corpus,
    *,
    model: str | None = None,
    batch_size: int = 32,
    settings: KaosMLCoreSettings | None = None,
) -> np.ndarray:
    """Produce a dense feature matrix for a Corpus via kaos-nlp-transformers.

    Args:
        corpus: A Corpus built from one or more ContentDocuments.
        model: Embedding model id. Defaults to
            ``settings.default_embed_model`` (``BAAI/bge-small-en-v1.5``).
        batch_size: Inference batch size passed to the backend.
        settings: Module settings; defaults to
            ``KaosMLCoreSettings.resolve(None)`` (env-resolved).

    Returns:
        A float32 numpy array of shape ``(len(corpus), embedding_dim)``.
        Row ``i`` corresponds to ``corpus.unit(i)``; the row index is
        the AST-grounded row index defined by the Corpus, satisfying
        invariant 4 in PRD §5.

    Raises:
        FeatureError: If the ``[transformers]`` extra is not installed,
            or if the backend returns an unexpected shape.
    """
    try:
        transformers = importlib.import_module("kaos_nlp_transformers")
    except ImportError as exc:
        msg = (
            "embed_corpus requires the [transformers] extra. "
            "Fix: install kaos-ml-core[transformers] (which pulls in "
            "kaos-nlp-transformers and fastembed). "
            "Alternative: in v1.1 a sparse TF-IDF feature path will be "
            "available via kaos_ml_core.features.tfidf_corpus()."
        )
        raise FeatureError(msg) from exc

    EmbeddingModel = transformers.EmbeddingModel
    resolved = KaosMLCoreSettings.resolve(settings)
    model_id = model or resolved.default_embed_model
    em = EmbeddingModel.load(model_id)
    texts = [u.text for u in corpus]
    vecs = em.embed(texts, batch_size=batch_size)

    if vecs.shape[0] != len(corpus):
        msg = (
            f"Embedding model returned {vecs.shape[0]} vectors for "
            f"{len(corpus)} corpus units. "
            "Fix: this is a kaos-nlp-transformers bug — file an issue "
            f"with the model id {model_id!r}."
        )
        raise FeatureError(msg)

    return vecs.astype(np.float32, copy=False)


def tfidf_corpus(corpus: Corpus, **kwargs: Any) -> None:
    """Produce a sparse TF-IDF feature matrix for a Corpus.

    Not implemented in v0 — lands in Phase v1.1, will consume
    kaos-nlp-core's existing InvertedIndex / SparseTermMatrix and emit
    a scipy.sparse.csr_matrix. See
    ``docs/internal/plans/kaos-ml-core-v0.md`` Phase v1.1.
    """
    msg = (
        "tfidf_corpus is not implemented in v0. "
        "Fix: use embed_corpus(corpus) for dense features in v0. "
        "Alternative: wait for Phase v1.1 which adds the sparse path."
    )
    raise NotImplementedError(msg)


__all__ = ["embed_corpus", "tfidf_corpus"]
