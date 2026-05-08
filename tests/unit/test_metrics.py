"""Unit tests for kaos_ml_core.metrics — Wilson CI + evaluate()."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kaos_ml_core.metrics import Metrics, evaluate, wilson_score_interval

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# wilson_score_interval — boundary & accuracy
# -----------------------------------------------------------------------------


class TestWilsonScoreInterval:
    def test_zero_total_returns_max_uncertainty(self):
        lo, hi = wilson_score_interval(0, 0)
        assert lo == 0.0 and hi == 1.0

    def test_zero_positives(self):
        lo, hi = wilson_score_interval(0, 100, confidence=0.95)
        assert lo == 0.0
        assert 0.0 < hi < 0.05  # tight upper bound near 0

    def test_all_positives(self):
        lo, hi = wilson_score_interval(100, 100, confidence=0.95)
        assert hi == 1.0
        assert 0.95 < lo < 1.0  # tight lower bound near 1

    def test_balanced_50_50(self):
        # 50/100 should give a symmetric-ish interval around 0.5
        lo, hi = wilson_score_interval(50, 100, confidence=0.95)
        center = (lo + hi) / 2
        assert abs(center - 0.5) < 0.01

    def test_known_reference_point(self):
        # 80 of 100 successes at 95% confidence should give Wilson CI
        # ≈ [0.711, 0.867] per the standard tables.
        lo, hi = wilson_score_interval(80, 100, confidence=0.95)
        assert abs(lo - 0.711) < 0.005
        assert abs(hi - 0.867) < 0.005

    def test_99_percent_wider_than_95(self):
        lo95, hi95 = wilson_score_interval(50, 100, confidence=0.95)
        lo99, hi99 = wilson_score_interval(50, 100, confidence=0.99)
        # 99% confidence interval must be at least as wide as 95%.
        assert (hi99 - lo99) > (hi95 - lo95)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match=r"confidence"):
            wilson_score_interval(50, 100, confidence=0.0)
        with pytest.raises(ValueError, match=r"confidence"):
            wilson_score_interval(50, 100, confidence=1.0)

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match=r"≥ 0"):
            wilson_score_interval(-1, 100)
        with pytest.raises(ValueError, match=r"≥ 0"):
            wilson_score_interval(50, -10)

    def test_positives_exceed_total_raises(self):
        with pytest.raises(ValueError, match=r"cannot exceed"):
            wilson_score_interval(110, 100)


# -----------------------------------------------------------------------------
# evaluate — binary, multi-class, edge cases
# -----------------------------------------------------------------------------


class TestEvaluateBinary:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = y_true.copy()
        m = evaluate(y_true, y_pred, classes=[0, 1])
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.accuracy == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = 1 - y_true
        m = evaluate(y_true, y_pred, classes=[0, 1])
        assert m.precision == 0.0
        assert m.recall == 0.0

    def test_includes_wilson_recall_ci(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        y_pred = y_true.copy()
        # Inject 10% errors
        flip = rng.choice(200, 20, replace=False)
        y_pred[flip] = 1 - y_pred[flip]
        m = evaluate(y_true, y_pred, classes=[0, 1])
        assert 0.0 <= m.recall_ci_lower <= m.recall <= m.recall_ci_upper <= 1.0
        assert m.confidence == 0.95

    def test_string_labels(self):
        y_true = np.array(["neg"] * 5 + ["pos"] * 5)
        y_pred = y_true.copy()
        m = evaluate(y_true, y_pred, classes=("neg", "pos"))
        assert m.precision == 1.0
        assert m.classes == ("neg", "pos")

    def test_roc_auc_with_proba(self):
        # Strong-signal classifier — proba should give AUC ≈ 1.0
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.where(y_true == 1, 0.9, 0.1)
        y_pred = (y_proba >= 0.5).astype(int)
        m = evaluate(y_true, y_pred, y_proba=y_proba, classes=[0, 1])
        assert m.roc_auc is not None
        assert m.roc_auc > 0.99

    def test_roc_auc_pos_class_correctly_identified(self):
        # If pos_label is the alphabetically-LATER class but proba is the
        # probability of being positive, AUC must still be ~1.0 (not 0.0).
        # Pins the explicit pos-class binarization fix.
        y_true = np.array(["a"] * 50 + ["z"] * 50)  # 'z' is positive
        y_proba = np.where(y_true == "z", 0.9, 0.1)
        y_pred = np.where(y_proba >= 0.5, "z", "a")
        m = evaluate(y_true, y_pred, y_proba=y_proba, classes=("a", "z"))
        assert m.roc_auc is not None and m.roc_auc > 0.99, f"ROC AUC inverted: got {m.roc_auc}"

    def test_no_proba_no_auc(self):
        y = np.array([0, 1, 0, 1])
        m = evaluate(y, y, classes=[0, 1])
        assert m.roc_auc is None


class TestEvaluateMultiClass:
    def test_three_class_macro_average(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = y_true.copy()
        m = evaluate(y_true, y_pred, classes=[0, 1, 2])
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.classes == ("0", "1", "2")

    def test_multi_class_no_roc_auc(self):
        # Multi-class ROC AUC has known sklearn quirks; we explicitly
        # skip it for binary contract clarity.
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = y_true.copy()
        # y_proba would be (n, 3) for multi-class; we only support binary.
        m = evaluate(y_true, y_pred, classes=[0, 1, 2])
        assert m.roc_auc is None


class TestEvaluateEdgeCases:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match=r"shape"):
            evaluate(np.array([0, 1]), np.array([0]), classes=[0, 1])

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match=r"1-D"):
            evaluate(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]), classes=[0, 1])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match=r"at least one row"):
            evaluate(np.array([]), np.array([]), classes=[0, 1])

    def test_proba_shape_mismatch_raises(self):
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match=r"y_proba shape"):
            evaluate(y, y, y_proba=np.array([0.5, 0.5]), classes=[0, 1])


class TestMetricsSerialization:
    def test_metrics_is_immutable(self):
        m = Metrics(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            accuracy=1.0,
            support=10,
            n_total=10,
            recall_ci_lower=0.7,
            recall_ci_upper=1.0,
            confidence=0.95,
            confusion_matrix=((5, 0), (0, 5)),
            classes=("neg", "pos"),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.precision = 0.5  # type: ignore[misc]  # ty: ignore[invalid-assignment]

    def test_metrics_carries_threshold(self):
        m = Metrics(
            precision=0.9,
            recall=0.8,
            f1=0.85,
            accuracy=0.85,
            support=100,
            n_total=100,
            recall_ci_lower=0.7,
            recall_ci_upper=0.9,
            confidence=0.95,
            confusion_matrix=((50, 5), (15, 30)),
            classes=("0", "1"),
            threshold=0.65,
        )
        assert m.threshold == 0.65
