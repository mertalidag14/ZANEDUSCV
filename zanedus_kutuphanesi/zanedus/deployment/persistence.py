"""Python'un kendi pickle modulu ile kaydetme ve yukleme."""

import pickle
from pathlib import Path


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(path):
    """Uyari: pickle dosyalari sadece guvendigin kaynaktan yuklenmeli."""
    with open(path, "rb") as f:
        return pickle.load(f)
