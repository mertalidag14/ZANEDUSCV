"""Olceklendirme."""

import numpy as np

from ..base import BaseEstimator, TransformerMixin


def _safe(scale):
    scale = np.asarray(scale, dtype=float)
    scale[scale == 0] = 1.0
    return scale


class StandardScaler(BaseEstimator, TransformerMixin):
    """(x - ortalama) / standart sapma"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = _safe(X.std(axis=0))
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Degerleri 0-1 arasina sikistirir."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.scale_ = _safe(X.max(axis=0) - self.min_)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.min_) / self.scale_


class RobustScaler(BaseEstimator, TransformerMixin):
    """(x - medyan) / IQR. Aykiri degerlerden az etkilenir."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.center_ = np.median(X, axis=0)
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        self.scale_ = _safe(q3 - q1)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.center_) / self.scale_


SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
