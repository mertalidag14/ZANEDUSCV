"""Aykiri deger tespiti ve kirpma."""

import numpy as np

from ..base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    """IQR ya da z-skoru ile aykiri degerleri sinirlara kirpar.

    method="iqr": Q1 - k*IQR ve Q3 + k*IQR disindakileri kirpar (k varsayilan 1.5).
    method="zscore": ortalamadan k standart sapma uzaktakileri kirpar (k varsayilan 3).
    """

    def __init__(self, method="iqr", k=None):
        self.method = method
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.method == "iqr":
            k = 1.5 if self.k is None else self.k
            q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
            iqr = q3 - q1
            self.lower_, self.upper_ = q1 - k * iqr, q3 + k * iqr
        elif self.method == "zscore":
            k = 3.0 if self.k is None else self.k
            mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
            self.lower_, self.upper_ = mean - k * std, mean + k * std
        else:
            raise ValueError("method 'iqr' ya da 'zscore' olmali")
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def outlier_mask(self, X):
        """Hangi hucrelerin aykiri oldugunu True/False olarak dondurur."""
        X = np.asarray(X, dtype=float)
        return (X < self.lower_) | (X > self.upper_)
