"""Send requests to a running service and report latency.

    python scripts/load_test.py --url http://127.0.0.1:8000 --n 1000
"""
import argparse
import json
import statistics
import time

import httpx
from sklearn.datasets import load_wine

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--batch", type=int, default=1)
args = ap.parse_args()

X, _ = load_wine(return_X_y=True)
lat = []
with httpx.Client(base_url=args.url, timeout=10) as c:
    c.get("/health").raise_for_status()
    t0 = time.perf_counter()
    for i in range(args.n):
        rows = [X[(i + k) % len(X)].tolist() for k in range(args.batch)]
        t = time.perf_counter()
        c.post("/predict", json={"instances": rows}).raise_for_status()
        lat.append((time.perf_counter() - t) * 1000)
    total = time.perf_counter() - t0
lat.sort()
print(json.dumps(dict(requests=args.n, batch=args.batch, throughput_rps=round(args.n / total, 1),
                      p50_ms=round(statistics.median(lat), 2), p95_ms=round(lat[int(0.95 * len(lat))], 2),
                      p99_ms=round(lat[int(0.99 * len(lat))], 2)), indent=2))
