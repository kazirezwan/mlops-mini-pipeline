import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_wine

from mlpipe import config, train


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    mp = pytest.MonkeyPatch()
    for name, value in dict(MODEL_DIR=d, MODEL_PATH=d / "model.joblib", METADATA_PATH=d / "metadata.json",
                            REFERENCE_PATH=d / "reference.npz").items():
        mp.setattr(config, name, value)
    train.train()
    from mlpipe.app import app
    with TestClient(app) as c:
        yield c
    mp.undo()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["metrics"]["test_accuracy"] >= 0.9
    assert info["class_names"] == ["class_0", "class_1", "class_2"]


def test_predict_matches_labels(client):
    X, y = load_wine(return_X_y=True)
    r = client.post("/predict", json={"instances": X[:20].tolist()})
    assert r.status_code == 200
    preds = [p["label"] for p in r.json()["predictions"]]
    assert np.mean(np.array(preds) == y[:20]) >= 0.9
    assert abs(sum(r.json()["predictions"][0]["probabilities"].values()) - 1) < 1e-3


@pytest.mark.parametrize("bad", [{"instances": [[1.0] * 12]}, {"instances": []}, {"rows": [[1.0] * 13]},
                                 {"instances": [["a"] * 13]}])
def test_predict_rejects_bad_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_drift_endpoint(client):
    X, _ = load_wine(return_X_y=True)
    same = client.post("/drift", json={"instances": X.tolist()}).json()
    shifted = X.copy()
    shifted[:, 0] += 2.0                       # alcohol +2 %
    moved = client.post("/drift", json={"instances": shifted.tolist()}).json()
    assert same["drift"] in ("none", "moderate")
    assert moved["drift"] == "major" and moved["worst_feature"] == "alcohol"


def test_metrics_endpoint(client):
    text = client.get("/metrics").text
    assert "mlpipe_rows_predicted_total" in text and 'quantile="0.95"' in text
