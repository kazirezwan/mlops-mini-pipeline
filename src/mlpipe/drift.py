"""Population Stability Index (PSI) for detecting input drift.

Rule of thumb: PSI < 0.1 no change, 0.1-0.25 moderate shift, > 0.25 major shift.
"""
from __future__ import annotations

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference: np.ndarray, current: np.ndarray, feature_names: list[str]) -> dict:
    scores = {f: round(psi(reference[:, j], current[:, j]), 4) for j, f in enumerate(feature_names)}
    worst = max(scores, key=scores.get)
    level = "major" if scores[worst] > 0.25 else "moderate" if scores[worst] > 0.1 else "none"
    return dict(psi=scores, worst_feature=worst, worst_psi=scores[worst], drift=level)
