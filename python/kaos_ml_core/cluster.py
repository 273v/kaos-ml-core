"""Clustering for a feature matrix and cold-start row selection.

v0: one algorithm only — ``MiniBatchKMeans`` on L2-normalized inputs
(spherical k-means). HDBSCAN + c-TF-IDF cluster labels land in Phase v1.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

__all__ = ["ClusterResult", "kmedoid_seeds", "minibatch_kmeans"]


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Output of a clustering run, with row-aligned labels.

    ``labels[i]`` is the cluster id assigned to row ``i`` of the input
    feature matrix — equivalently, the cluster id for ``corpus.unit(i)``.
    """

    labels: np.ndarray
    """Shape ``(N,)``, dtype int. Cluster id per input row."""

    centroids: np.ndarray
    """Shape ``(n_clusters, D)``. Final cluster centroids."""

    n_clusters: int
    """Effective number of clusters after fit."""

    inertia: float
    """Final inertia (sum of squared distances to centroids)."""


def minibatch_kmeans(
    X: np.ndarray,
    *,
    n_clusters: int = 20,
    random_state: int = 0,
    batch_size: int = 1024,
    n_init: str | int = "auto",
) -> ClusterResult:
    """Run spherical MiniBatchKMeans on L2-normalized inputs.

    L2-normalizing before k-means turns Euclidean distance into cosine
    distance, which is the right similarity for text embeddings. This
    is the v0 default and the consensus 2026 choice for short-document
    clustering on dense embeddings.

    Args:
        X: Feature matrix, shape ``(N, D)``.
        n_clusters: Number of clusters to fit.
        random_state: Random seed for determinism.
        batch_size: Size of MiniBatch updates.
        n_init: Number of initializations; ``"auto"`` lets sklearn pick.

    Returns:
        ClusterResult with row-aligned labels.
    """
    if X.ndim != 2:
        msg = (
            f"X must be 2-D, got shape {X.shape}. "
            "Fix: produce X via embed_corpus() or any (N, D) array."
        )
        raise ValueError(msg)

    if n_clusters < 1:
        msg = f"n_clusters must be >= 1, got {n_clusters}"
        raise ValueError(msg)

    Xn = normalize(X, norm="l2", axis=1)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
    )
    labels = km.fit_predict(Xn)
    return ClusterResult(
        labels=np.asarray(labels, dtype=np.int64),
        centroids=np.asarray(km.cluster_centers_, dtype=np.float32),
        n_clusters=int(km.n_clusters),
        inertia=float(km.inertia_),
    )


def kmedoid_seeds(
    X: np.ndarray,
    result: ClusterResult,
    *,
    per_cluster: int = 3,
) -> list[int]:
    """Pick the rows closest to each cluster centroid.

    For each cluster, returns the ``per_cluster`` rows whose feature
    vectors are closest (cosine similarity) to the cluster centroid.
    These are the most representative documents per cluster, which is
    the TAR-consensus first-batch strategy when no model exists yet.

    The returned list is the cold-start seed set for LLM labeling —
    pass it to ``label_seeds_with_llm(corpus, seeds, ...)``.

    Args:
        X: Feature matrix, shape ``(N, D)``. Must match ``result.labels``.
        result: ClusterResult from ``minibatch_kmeans()``.
        per_cluster: Number of seeds per cluster.

    Returns:
        Sorted list of unique row indices.
    """
    if per_cluster < 1:
        msg = f"per_cluster must be >= 1, got {per_cluster}"
        raise ValueError(msg)
    if X.shape[0] != result.labels.shape[0]:
        msg = (
            f"X has {X.shape[0]} rows but labels have {result.labels.shape[0]}. "
            "Fix: pass the same feature matrix that produced the cluster result."
        )
        raise ValueError(msg)

    Xn = normalize(X, norm="l2", axis=1)
    seeds: list[int] = []

    for cluster_id in range(result.n_clusters):
        rows_in_cluster = np.where(result.labels == cluster_id)[0]
        if len(rows_in_cluster) == 0:
            continue
        centroid = result.centroids[cluster_id]
        sims = Xn[rows_in_cluster] @ centroid
        order = np.argsort(-sims)[:per_cluster]
        seeds.extend(int(rows_in_cluster[i]) for i in order)

    return sorted(set(seeds))
