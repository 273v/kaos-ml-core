"""Unit tests for kaos_ml_core.pipeline — Pipeline save/load + persistence."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from kaos_ml_core import Metrics, Pipeline, PipelineError

pytestmark = pytest.mark.unit


def _fitted_classifier(seed: int = 0):
    """Tiny LogisticRegression fitted on synthetic 2-class data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((100, 16))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int).astype(str)
    return LogisticRegression().fit(X, y), X, y


def _make_pipeline(**overrides) -> Pipeline:
    clf, _, _ = _fitted_classifier(seed=42)
    defaults: dict = {
        "embed_model_id": "BAAI/bge-small-en-v1.5",
        "embed_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "classifier": clf,
        "threshold": 0.5,
        "classes": ("0", "1"),
        "kaos_ml_core_version": "0.1.0a1-test",
    }
    defaults.update(overrides)
    return Pipeline(**defaults)


class TestPipelineSaveLoad:
    def test_round_trip_basic(self, tmp_path: Path):
        p = _make_pipeline()
        saved = p.save(tmp_path / "test.kaos")
        assert saved.is_dir()
        assert (saved / "manifest.json").is_file()
        assert (saved / "classifier.joblib").is_file()

        loaded = Pipeline.load(saved)
        assert loaded.embed_model_id == p.embed_model_id
        assert loaded.embed_revision == p.embed_revision
        assert loaded.threshold == p.threshold
        assert loaded.classes == p.classes
        assert loaded.kaos_ml_core_version == p.kaos_ml_core_version

    def test_classifier_round_trips(self, tmp_path: Path):
        clf, X, _ = _fitted_classifier()
        p = _make_pipeline(classifier=clf)
        saved = p.save(tmp_path / "rt.kaos")
        loaded = Pipeline.load(saved)
        # Same predictions on the original data.
        assert np.allclose(
            p.classifier.predict_proba(X),
            loaded.classifier.predict_proba(X),
        )

    def test_train_metrics_round_trip(self, tmp_path: Path):
        m = Metrics(
            precision=0.91,
            recall=0.86,
            f1=0.88,
            accuracy=0.92,
            support=100,
            n_total=200,
            recall_ci_lower=0.78,
            recall_ci_upper=0.92,
            confidence=0.95,
            confusion_matrix=((90, 10), (14, 86)),
            classes=("0", "1"),
            threshold=0.55,
            roc_auc=0.94,
        )
        p = _make_pipeline(train_metrics=m)
        saved = p.save(tmp_path / "rt-metrics.kaos")
        loaded = Pipeline.load(saved)
        assert loaded.train_metrics is not None
        assert abs(loaded.train_metrics.precision - 0.91) < 1e-9
        assert loaded.train_metrics.classes == ("0", "1")
        assert loaded.train_metrics.confusion_matrix == ((90, 10), (14, 86))


class TestPipelineLoadValidation:
    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(PipelineError, match=r"is not a directory"):
            Pipeline.load(tmp_path / "does-not-exist")

    def test_path_is_file_not_dir_raises(self, tmp_path: Path):
        f = tmp_path / "not-a-dir.kaos"
        f.write_text("not a directory")
        with pytest.raises(PipelineError, match=r"is not a directory"):
            Pipeline.load(f)

    def test_missing_manifest_raises(self, tmp_path: Path):
        d = tmp_path / "no-manifest.kaos"
        d.mkdir()
        (d / "classifier.joblib").write_bytes(b"junk")
        with pytest.raises(PipelineError, match=r"Manifest not found"):
            Pipeline.load(d)

    def test_missing_classifier_raises(self, tmp_path: Path):
        p = _make_pipeline()
        saved = p.save(tmp_path / "no-clf.kaos")
        (saved / "classifier.joblib").unlink()
        with pytest.raises(PipelineError, match=r"Classifier not found"):
            Pipeline.load(saved)

    def test_magic_byte_rejection(self, tmp_path: Path):
        p = _make_pipeline()
        saved = p.save(tmp_path / "magic.kaos")
        # Tamper with the manifest's magic string.
        manifest = json.loads((saved / "manifest.json").read_text())
        manifest["_kaos_pipeline_format"] = "evil/spoofed:v0"
        (saved / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(PipelineError, match=r"magic string"):
            Pipeline.load(saved)

    def test_unknown_format_version_rejection(self, tmp_path: Path):
        p = _make_pipeline()
        saved = p.save(tmp_path / "format.kaos")
        manifest = json.loads((saved / "manifest.json").read_text())
        manifest["format_version"] = 999  # future version
        (saved / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(PipelineError, match=r"format_version"):
            Pipeline.load(saved)

    def test_corrupt_manifest_json_rejection(self, tmp_path: Path):
        p = _make_pipeline()
        saved = p.save(tmp_path / "corrupt.kaos")
        (saved / "manifest.json").write_text("{not json")
        with pytest.raises(PipelineError, match=r"valid JSON"):
            Pipeline.load(saved)


class TestPipelineImmutability:
    def test_pipeline_is_frozen(self):
        p = _make_pipeline()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.threshold = 0.99  # type: ignore[misc]  # ty: ignore[invalid-assignment]


class TestPipelineExtras:
    def test_extras_persist(self, tmp_path: Path):
        p = _make_pipeline(extras={"trained_by": "alice", "case_id": 42})
        saved = p.save(tmp_path / "extras.kaos")
        loaded = Pipeline.load(saved)
        assert loaded.extras == {"trained_by": "alice", "case_id": 42}

    def test_non_serializable_extras_raises(self, tmp_path: Path):
        p = _make_pipeline(extras={"bad": np.array([1, 2, 3])})
        with pytest.raises(PipelineError, match=r"JSON-serializable"):
            p.save(tmp_path / "bad-extras.kaos")
