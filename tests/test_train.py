import json

import numpy as np
import pytest

from mlpipe import config, train
from mlpipe.drift import psi


def test_train_writes_versioned_artifacts(tmp_path, monkeypatch):
    for name, value in dict(MODEL_DIR=tmp_path, MODEL_PATH=tmp_path / "model.joblib",
                            METADATA_PATH=tmp_path / "metadata.json",
                            REFERENCE_PATH=tmp_path / "reference.npz").items():
        monkeypatch.setattr(config, name, value)
    meta = train.train()
    assert (tmp_path / "model.joblib").exists() and (tmp_path / "reference.npz").exists()
    assert json.loads((tmp_path / "metadata.json").read_text())["model_version"] == meta["model_version"]
    assert meta["quality_gate"]["passed"] and len(meta["feature_names"]) == 13


def test_quality_gate_blocks_weak_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    with pytest.raises(train.QualityGateError):
        train.train(min_accuracy=1.01)
    assert not (tmp_path / "model.joblib").exists()


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=2000)
    assert psi(ref, rng.normal(size=500)) < 0.1
    assert psi(ref, rng.normal(1.0, 1, size=500)) > 0.25
