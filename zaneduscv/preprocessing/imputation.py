"""Eksik veri doldurma."""

import numpy as np

from ..base import BaseEstimator, TransformerMixin


def _is_missing(X):
    X = np.asarray(X, dtype=object)
    out = np.zeros(X.shape, dtype=bool)
    for idx, v in np.ndenumerate(X):
        out[idx] = v is None or (isinstance(v, float) and np.isnan(v))
    return out


class SimpleImputer(BaseEstimator, TransformerMixin):
    """strategy: "mean", "median", "most_frequent", "constant"."""

    def __init__(self, strategy="median", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        missing = _is_missing(X)
        self.statistics_ = []
        for j in range(X.shape[1]):
            col = X[~missing[:, j], j]
            if self.strategy == "constant":
                value = self.fill_value if self.fill_value is not None else 0
            elif len(col) == 0:
                value = 0.0 if self.strategy in ("mean", "median") else "eksik"
            elif self.strategy == "mean":
                value = float(np.mean(col.astype(float)))
            elif self.strategy == "median":
                value = float(np.median(col.astype(float)))
            elif self.strategy == "most_frequent":
                values, counts = np.unique(col.astype(str), return_counts=True)
                best = values[np.argmax(counts)]
                # orijinal tipi koru (sayi ise sayi olarak dondur)
                value = next(v for v in col if str(v) == best)
            else:
                raise ValueError(f"Gecersiz strateji: {self.strategy}")
            self.statistics_.append(value)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        missing = _is_missing(X)
        for j, value in enumerate(self.statistics_):
            X[missing[:, j], j] = value
        return X


class KNNImputer(BaseEstimator, TransformerMixin):
    """Eksik degeri, en yakin k komsunun ortalamasiyla doldurur (sadece sayisal)."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.train_ = np.asarray(X, dtype=float)
        self.col_means_ = np.nan_to_num(np.nanmean(self.train_, axis=0))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        train = self.train_
        for i in np.where(np.isnan(X).any(axis=1))[0]:
            row = X[i]
            known = ~np.isnan(row)
            # sadece bilinen kolonlar uzerinden mesafe; eksikler kolon ortalamasi
            diff = np.where(np.isnan(train[:, known]), 0.0, train[:, known] - row[known])
            dist = np.sqrt((diff ** 2).sum(axis=1))
            for j in np.where(~known)[0]:
                candidates = np.where(~np.isnan(train[:, j]))[0]
                if len(candidates) == 0:
                    X[i, j] = self.col_means_[j]
                    continue
                nearest = candidates[np.argsort(dist[candidates])[: self.n_neighbors]]
                X[i, j] = train[nearest, j].mean()
        return X
