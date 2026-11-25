"""Train the model, check it against a quality gate and save a versioned artifact.

    python -m mlpipe.train
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlpipe import __version__, config


class QualityGateError(RuntimeError):
    pass


def data_fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()[:16]


def train(seed: int = 42, min_accuracy: float | None = None) -> dict:
    min_accuracy = config.MIN_ACCURACY if min_accuracy is None else min_accuracy
    data = load_wine()
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25,
                                              random_state=seed, stratify=data.target)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    cv = cross_val_score(model, X_tr, y_tr, cv=5)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = dict(test_accuracy=round(float(accuracy_score(y_te, pred)), 4),
                   test_macro_f1=round(float(f1_score(y_te, pred, average="macro")), 4),
                   cv_accuracy_mean=round(float(cv.mean()), 4), cv_accuracy_std=round(float(cv.std()), 4))
    if metrics["test_accuracy"] < min_accuracy:
        raise QualityGateError(f"test accuracy {metrics['test_accuracy']} < required {min_accuracy}")

    trained_at = datetime.now(timezone.utc)
    meta = dict(
        model_version=f"{__version__}+{trained_at.strftime('%Y%m%d%H%M%S')}",
        trained_at=trained_at.isoformat(timespec="seconds"),
        algorithm="StandardScaler + LogisticRegression",
        dataset="UCI Wine (sklearn.datasets.load_wine)",
        data_fingerprint=data_fingerprint(data.data, data.target),
        n_train=int(len(y_tr)), n_test=int(len(y_te)),
        feature_names=list(data.feature_names), class_names=list(data.target_names),
        metrics=metrics, quality_gate=dict(min_test_accuracy=min_accuracy, passed=True),
        python=platform.python_version(), sklearn=sklearn.__version__,
    )
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    config.METADATA_PATH.write_text(json.dumps(meta, indent=2))
    np.savez(config.REFERENCE_PATH, X=X_tr)            # reference sample for drift checks
    return meta


if __name__ == "__main__":
    try:
        print(json.dumps(train(), indent=2))
    except QualityGateError as e:
        print(f"QUALITY GATE FAILED: {e}", file=sys.stderr)
        sys.exit(1)
