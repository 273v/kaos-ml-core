"""Unit tests for kaos_ml_core.train — classifier training.

Tests use small synthetic data to verify train_classifier and train_logreg
produce fitted sklearn classifiers with the expected properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from kaos_ml_core.errors import TrainError
from kaos_ml_core.train import train_classifier, train_logreg

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def labeled_data() -> tuple[np.ndarray, dict[int, str]]:
    """A simple feature matrix with binary labels for rows 0-9."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 5)).astype(np.float32)
    # Label first 10 rows: half positive, half negative
    labels = {
        0: "relevant",
        1: "relevant",
        2: "relevant",
        3: "relevant",
        4: "relevant",
        5: "not_relevant",
        6: "not_relevant",
        7: "not_relevant",
        8: "not_relevant",
        9: "not_relevant",
    }
    return X, labels


# ---------------------------------------------------------------------------
# train_logreg
# ---------------------------------------------------------------------------


class TestTrainLogreg:
    def test_basic(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        clf = train_logreg(X, labels)
        assert hasattr(clf, "predict_proba")
        assert hasattr(clf, "classes_")
        assert set(clf.classes_) == {"relevant", "not_relevant"}

    def test_uses_liblinear(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        clf = train_logreg(X, labels)
        assert clf.solver == "liblinear"

    def test_balanced_class_weight(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        clf = train_logreg(X, labels)
        assert clf.class_weight == "balanced"

    def test_custom_C(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        clf = train_logreg(X, labels, C=0.1)
        assert pytest.approx(0.1) == clf.C

    def test_predict_proba_shape(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        clf = train_logreg(X, labels)
        proba = clf.predict_proba(X)
        assert proba.shape == (20, 2)

    def test_rejects_empty_labels(self) -> None:
        X = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(TrainError, match="at least one"):
            train_logreg(X, {})

    def test_rejects_single_class(self) -> None:
        X = np.zeros((5, 3), dtype=np.float32)
        labels = {0: "same", 1: "same", 2: "same"}
        with pytest.raises(TrainError, match="at least 2"):
            train_logreg(X, labels)


# ---------------------------------------------------------------------------
# train_classifier dispatcher
# ---------------------------------------------------------------------------


class TestTrainClassifier:
    def test_default_is_logreg(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        clf = train_classifier(X, labels)
        from sklearn.linear_model import LogisticRegression

        assert isinstance(clf, LogisticRegression)

    def test_explicit_logreg(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        clf = train_classifier(X, labels, model="logreg")
        assert hasattr(clf, "predict_proba")

    def test_rejects_rf(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        with pytest.raises(NotImplementedError, match="Random Forest"):
            train_classifier(X, labels, model="rf")

    def test_rejects_unknown_model(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        with pytest.raises(TrainError, match="Unknown model"):
            train_classifier(X, labels, model="xgboost")

    def test_passes_kwargs(self, labeled_data: tuple[np.ndarray, dict[int, str]]) -> None:
        X, labels = labeled_data
        clf = train_classifier(X, labels, model="logreg", C=10.0)
        assert pytest.approx(10.0) == clf.C
