"""K-en yakin komsu."""

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array


def _pairwise_sq_dist(A, B):
    d = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
    return np.maximum(d, 0)


class _KNNBase(BaseEstimator):
    def __init__(self, n_neighbors=5, weights="uniform", batch_size=1024):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_ = check_array(X)
        self.y_ = np.asarray(y).ravel()
        return self

    def _neighbors(self, X):
        X = check_array(X)
        k = min(self.n_neighbors, len(self.X_))
        idx_all, dist_all = [], []
        for start in range(0, len(X), self.batch_size):
            d = _pairwise_sq_dist(X[start:start + self.batch_size], self.X_)
            idx = np.argpartition(d, k - 1, axis=1)[:, :k]
            idx_all.append(idx)
            dist_all.append(np.sqrt(np.take_along_axis(d, idx, axis=1)))
        idx, dist = np.vstack(idx_all), np.vstack(dist_all)
        if self.weights == "distance":
            w = 1.0 / (dist + 1e-9)
        else:
            w = np.ones_like(dist)
        return idx, w


class KNNClassifier(_KNNBase, ClassifierMixin):
    def fit(self, X, y):
        super().fit(X, y)
        self.y_ = self.y_.astype(int)
        self.n_classes_ = int(self.y_.max()) + 1
        return self

    def predict_proba(self, X):
        idx, w = self._neighbors(X)
        labels = self.y_[idx]
        proba = np.zeros((len(idx), self.n_classes_))
        for c in range(self.n_classes_):
            proba[:, c] = (w * (labels == c)).sum(axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class KNNRegressor(_KNNBase, RegressorMixin):
    def predict(self, X):
        idx, w = self._neighbors(X)
        y = self.y_.astype(float)[idx]
        return (w * y).sum(axis=1) / w.sum(axis=1)
