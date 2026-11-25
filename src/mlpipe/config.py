import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
REFERENCE_PATH = MODEL_DIR / "reference.npz"
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.90"))   # quality gate
