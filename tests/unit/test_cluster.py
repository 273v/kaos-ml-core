"""Unit tests for kaos_ml_core.cluster — MiniBatchKMeans and kmedoid seeds.

Tests use small synthetic numpy arrays to keep execution fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from kaos_ml_core.cluster import ClusterResult, kmedoid_seeds, minibatch_kmeans

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_matrix() -> np.ndarray:
    """A small (20, 4) feature matrix with 2 clear clusters."""
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=1.0, scale=0.1, size=(10, 4))
    cluster_b = rng.normal(loc=-1.0, scale=0.1, size=(10, 4))
    return np.vstack([cluster_a, cluster_b]).astype(np.float32)


# ---------------------------------------------------------------------------
# minibatch_kmeans
# ---------------------------------------------------------------------------


class TestMinibatchKmeans:
    def test_basic(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=2, random_state=42)
        assert isinstance(result, ClusterResult)
        assert result.n_clusters == 2
        assert result.labels.shape == (20,)
        assert result.centroids.shape == (2, 4)
        assert result.inertia >= 0.0

    def test_labels_are_valid(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=3, random_state=0)
        assert set(result.labels).issubset({0, 1, 2})

    def test_single_cluster(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=1, random_state=0)
        assert result.n_clusters == 1
        assert np.all(result.labels == 0)

    def test_deterministic(self, simple_matrix: np.ndarray) -> None:
        r1 = minibatch_kmeans(simple_matrix, n_clusters=2, random_state=99)
        r2 = minibatch_kmeans(simple_matrix, n_clusters=2, random_state=99)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_rejects_1d_input(self) -> None:
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2-D"):
            minibatch_kmeans(X, n_clusters=2)

    def test_rejects_zero_clusters(self, simple_matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="n_clusters"):
            minibatch_kmeans(simple_matrix, n_clusters=0)

    def test_rejects_negative_clusters(self, simple_matrix: np.ndarray) -> None:
        with pytest.raises(ValueError, match="n_clusters"):
            minibatch_kmeans(simple_matrix, n_clusters=-1)

    def test_custom_batch_size(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=2, batch_size=5)
        assert result.labels.shape == (20,)

    def test_labels_dtype(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=2)
        assert result.labels.dtype == np.int64

    def test_centroids_dtype(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=2)
        assert result.centroids.dtype == np.float32


# ---------------------------------------------------------------------------
# kmedoid_seeds
# ---------------------------------------------------------------------------


class TestKmedoidSeeds:
    def test_basic(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=2, random_state=42)
        seeds = kmedoid_seeds(simple_matrix, result, per_cluster=2)
        assert isinstance(seeds, list)
        assert len(seeds) <= 4  # 2 clusters x 2 per_cluster
        assert all(isinstance(s, int) for s in seeds)
        assert all(0 <= s < 20 for s in seeds)

    def test_seeds_are_sorted_and_unique(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=2, random_state=42)
        seeds = kmedoid_seeds(simple_matrix, result, per_cluster=3)
        assert seeds == sorted(set(seeds))

    def test_per_cluster_one(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=2, random_state=42)
        seeds = kmedoid_seeds(simple_matrix, result, per_cluster=1)
        assert len(seeds) == 2  # one seed per cluster

    def test_rejects_zero_per_cluster(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=2, random_state=42)
        with pytest.raises(ValueError, match="per_cluster"):
            kmedoid_seeds(simple_matrix, result, per_cluster=0)

    def test_rejects_shape_mismatch(self, simple_matrix: np.ndarray) -> None:
        result = minibatch_kmeans(simple_matrix, n_clusters=2, random_state=42)
        bad_X = simple_matrix[:5]
        with pytest.raises(ValueError, match="rows"):
            kmedoid_seeds(bad_X, result, per_cluster=1)

    def test_per_cluster_exceeds_cluster_size(self, simple_matrix: np.ndarray) -> None:
        """When per_cluster > cluster size, should still work without error."""
        result = minibatch_kmeans(simple_matrix, n_clusters=2, random_state=42)
        seeds = kmedoid_seeds(simple_matrix, result, per_cluster=100)
        # Should return all rows since per_cluster exceeds each cluster
        assert len(seeds) <= 20
        assert len(seeds) > 0
