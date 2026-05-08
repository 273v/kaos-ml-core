"""Unit tests for kaos_ml_core.split — stratified train/test/control split."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kaos_ml_core import stratified_split

pytestmark = pytest.mark.unit


class TestStratifiedSplit:
    def test_three_way_split_disjoint_and_complete(self):
        rng = np.random.default_rng(0)
        labels = rng.choice(["a", "b"], size=1000, p=[0.7, 0.3])
        split = stratified_split(labels, test_frac=0.2, control_frac=0.1, seed=42)
        union = np.concatenate([split.train_idx, split.test_idx, split.control_idx])
        assert len(np.unique(union)) == 1000  # disjoint
        assert union.size == 1000  # complete
        assert split.n_train + split.n_test + split.n_control == 1000

    def test_class_ratio_preserved(self):
        rng = np.random.default_rng(0)
        labels = rng.choice(["a", "b"], size=1000, p=[0.85, 0.15])
        split = stratified_split(labels, test_frac=0.2, control_frac=0.1, seed=42)
        # Within ±2% of overall class balance.
        overall_pos = float(np.mean(labels == "b"))
        for idx in (split.train_idx, split.test_idx, split.control_idx):
            split_pos = float(np.mean(labels[idx] == "b"))
            assert abs(split_pos - overall_pos) < 0.02

    def test_control_frac_zero(self):
        labels = np.array(["a"] * 70 + ["b"] * 30)
        split = stratified_split(labels, test_frac=0.25, control_frac=0.0, seed=42)
        assert split.n_control == 0
        assert split.n_train + split.n_test == 100

    def test_deterministic_with_seed(self):
        labels = np.array(["a"] * 50 + ["b"] * 50)
        s1 = stratified_split(labels, test_frac=0.2, control_frac=0.1, seed=42)
        s2 = stratified_split(labels, test_frac=0.2, control_frac=0.1, seed=42)
        assert np.array_equal(s1.train_idx, s2.train_idx)
        assert np.array_equal(s1.test_idx, s2.test_idx)
        assert np.array_equal(s1.control_idx, s2.control_idx)

    def test_smallest_class_too_small_raises(self):
        # Only 2 'b' rows — can't split 3 ways
        labels = np.array(["a"] * 100 + ["b", "b"])
        with pytest.raises(ValueError, match=r"smallest class"):
            stratified_split(labels, test_frac=0.2, control_frac=0.1)

    def test_invalid_test_frac_raises(self):
        labels = np.array(["a"] * 50 + ["b"] * 50)
        with pytest.raises(ValueError, match=r"test_frac"):
            stratified_split(labels, test_frac=0.0)
        with pytest.raises(ValueError, match=r"test_frac"):
            stratified_split(labels, test_frac=1.0)

    def test_fractions_sum_too_high_raises(self):
        labels = np.array(["a"] * 50 + ["b"] * 50)
        with pytest.raises(ValueError, match=r"leave at least"):
            stratified_split(labels, test_frac=0.6, control_frac=0.5)

    def test_indices_are_sorted(self):
        labels = np.array(["a"] * 50 + ["b"] * 50)
        split = stratified_split(labels, test_frac=0.2, control_frac=0.1, seed=42)
        # Sorting makes downstream slicing deterministic.
        assert np.array_equal(split.train_idx, np.sort(split.train_idx))
        assert np.array_equal(split.test_idx, np.sort(split.test_idx))
        assert np.array_equal(split.control_idx, np.sort(split.control_idx))

    def test_split_result_is_immutable(self):
        labels = np.array(["a"] * 50 + ["b"] * 50)
        split = stratified_split(labels, test_frac=0.2, control_frac=0.1, seed=42)
        with pytest.raises(dataclasses.FrozenInstanceError):
            split.train_idx = np.array([])  # type: ignore[misc]  # ty: ignore[invalid-assignment]

    def test_result_n_properties(self):
        labels = np.array(["a"] * 50 + ["b"] * 50)
        split = stratified_split(labels, test_frac=0.2, control_frac=0.1, seed=42)
        assert split.n_train == split.train_idx.size
        assert split.n_test == split.test_idx.size
        assert split.n_control == split.control_idx.size
