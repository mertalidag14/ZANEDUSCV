"""Ortak yardimci fonksiyonlar."""

import numpy as np
import pandas as pd


def to_dataframe(X):
    """Gelen veriyi her zaman pandas DataFrame'e cevirir."""
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def detect_task(y):
    """Hedef degiskene bakip gorevin siniflandirma mi regresyon mu oldugunu tahmin eder."""
    y = pd.Series(np.asarray(y).ravel())
    if y.dtype == object or y.dtype.name in ("category", "bool"):
        return "classification"
    values = y.dropna()
    if pd.api.types.is_float_dtype(y) and not np.allclose(values, values.round()):
        return "regression"
    n_unique = values.nunique()
    if n_unique <= 20 or n_unique / max(len(values), 1) < 0.05:
        return "classification"
    return "regression"
