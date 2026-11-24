"""joblib ile kaydetme ve yukleme."""

from pathlib import Path

import joblib


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path):
    return joblib.load(path)
