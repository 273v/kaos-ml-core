"""Unit tests for kaos_ml_core.threshold — tune_threshold on control set."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from kaos_ml_core import tune_threshold

pytestmark = pytest.mark.unit


class TestTuneThresholdBinary:
    def _make_separable(self, n=200, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        y_true = rng.integers(0, 2, size=n)
        # Strong-signal classifier: positives sample from Beta(5, 1.5),
        # negatives from Beta(1.5, 5). ~85% recall achievable at sensible
        # thresholds.
        y_proba = np.where(y_true == 1, rng.beta(5, 1.5, n), rng.beta(1.5, 5, n))
        return y_true, y_proba

    def test_target_recall_is_met(self):
        y_true, y_proba = self._make_separable(n=500, rng_seed=42)
        res = tune_threshold(y_true, y_proba, target_recall=0.85)
        assert res.achieved_recall >= 0.85
        assert 0.0 < res.threshold < 1.0

    def test_target_precision_is_met(self):
        y_true, y_proba = self._make_separable(n=500, rng_seed=42)
        res = tune_threshold(y_true, y_proba, target_precision=0.90)
        assert res.achieved_precision >= 0.90

    def test_target_recall_picks_largest_threshold(self):
        # When multiple thresholds satisfy target_recall, we want the
        # LARGEST (highest precision). A small jitter around target should
        # show that picking-larger-than-needed pays off in precision.
        y_true, y_proba = self._make_separable(n=500, rng_seed=42)
        res_low = tune_threshold(y_true, y_proba, target_recall=0.50)
        res_high = tune_threshold(y_true, y_proba, target_recall=0.95)
        # Lower target → larger threshold → higher precision.
        assert res_low.threshold > res_high.threshold
        assert res_low.achieved_precision > res_high.achieved_precision

    def test_unattainable_precision_target_raises(self):
        # Construct a noisy classifier where max achievable precision
        # is below 0.99 (no threshold gets perfect precision because
        # negatives leak through). Recall=1.0 is always achievable at
        # threshold=0 — that's the invariant of precision_recall_curve.
        # So we test the precision branch where it's possible to have
        # NO threshold satisfy the target.
        rng = np.random.default_rng(123)
        # Heavily imbalanced + noisy: max precision is around 0.7.
        y_true = np.array([1] * 30 + [0] * 70)
        y_proba = np.concatenate(
            [
                rng.uniform(0.5, 1.0, size=30),  # positives uniform-high
                rng.uniform(0.4, 1.0, size=70),  # negatives also reach high scores
            ]
        )
        with pytest.raises(ValueError, match=r"No threshold achieves"):
            tune_threshold(y_true, y_proba, target_precision=0.99)


class TestTuneThresholdInputValidation:
    def test_neither_target_raises(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.3, 0.7, 0.4, 0.8])
        with pytest.raises(ValueError, match=r"requires exactly one"):
            tune_threshold(y_true, y_proba)

    def test_both_targets_raises(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.3, 0.7, 0.4, 0.8])
        with pytest.raises(ValueError, match=r"OR target_precision"):
            tune_threshold(y_true, y_proba, target_recall=0.5, target_precision=0.5)

    def test_proba_out_of_range_raises(self):
        y_true = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            tune_threshold(y_true, np.array([1.5, 0.7, 0.4, 0.8]), target_recall=0.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            tune_threshold(y_true, np.array([-0.1, 0.7, 0.4, 0.8]), target_recall=0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match=r"shape"):
            tune_threshold(np.array([0, 1]), np.array([0.5, 0.5, 0.5]), target_recall=0.5)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match=r"non-empty"):
            tune_threshold(np.array([]), np.array([]), target_recall=0.5)

    def test_invalid_target_range_raises(self):
        y_true = np.array([0, 1])
        y_proba = np.array([0.3, 0.7])
        with pytest.raises(ValueError, match=r"target_recall"):
            tune_threshold(y_true, y_proba, target_recall=0.0)
        with pytest.raises(ValueError, match=r"target_recall"):
            tune_threshold(y_true, y_proba, target_recall=1.5)


class TestHardPredictionWarning:
    def test_warns_on_hard_predictions(self):
        # 0.0 and 1.0 only — likely the user passed clf.predict() output
        # by mistake. Warn loudly per CLAUDE.md hard rule #5.
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_proba_hard = y_true.astype(float)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            tune_threshold(y_true, y_proba_hard, target_recall=0.5)
            assert len(captured) >= 1
            assert any(
                issubclass(w.category, RuntimeWarning)
                and "hard predictions" in str(w.message).lower()
                for w in captured
            )


class TestThresholdResult:
    def test_carries_target_back(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        res = tune_threshold(y_true, y_proba, target_recall=0.65)
        assert res.target_recall == 0.65
        assert res.target_precision is None
        assert res.n_control == 6

    def test_immutable(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.3, 0.7, 0.4, 0.8])
        res = tune_threshold(y_true, y_proba, target_recall=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.threshold = 0.99  # type: ignore[misc]  # ty: ignore[invalid-assignment]
