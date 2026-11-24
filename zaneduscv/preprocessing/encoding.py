"""Kategorik degiskenleri sayiya cevirme."""

import numpy as np

from ..base import BaseEstimator, TransformerMixin


class LabelEncoder(BaseEstimator):
    """Hedef etiketlerini 0..k-1 tam sayilara cevirir."""

    def __init__(self):
        pass

    def fit(self, y):
        self.classes_ = np.unique(np.asarray(y).ravel())
        return self

    def transform(self, y):
        y = np.asarray(y).ravel()
        idx = np.searchsorted(self.classes_, y)
        idx = np.clip(idx, 0, len(self.classes_) - 1)
        if not np.all(self.classes_[idx] == y):
            raise ValueError("Egitimde gorulmemis etiket var")
        return idx

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, idx):
        return self.classes_[np.asarray(idx, dtype=int)]


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Her kategoriye bir sayi verir. Bilinmeyen kategori -1 olur."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=str)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=str)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        out = np.full(X.shape, -1.0)
        for j, cats in enumerate(self.categories_):
            mapping = {c: i for i, c in enumerate(cats)}
            out[:, j] = [mapping.get(v, -1) for v in X[:, j]]
        return out


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Her kategori icin 0/1 kolonu acar. Cok seyrek kategoriler "diger" kolonunda toplanir."""

    def __init__(self, max_categories=30):
        self.max_categories = max_categories

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=str)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = []
        for j in range(X.shape[1]):
            values, counts = np.unique(X[:, j], return_counts=True)
            order = np.argsort(-counts, kind="stable")
            self.categories_.append(list(values[order][: self.max_categories]))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=str)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        blocks = []
        for j, cats in enumerate(self.categories_):
            block = np.zeros((X.shape[0], len(cats)))
            mapping = {c: i for i, c in enumerate(cats)}
            for i, v in enumerate(X[:, j]):
                if v in mapping:
                    block[i, mapping[v]] = 1.0
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.zeros((X.shape[0], 0))

    def get_feature_names_out(self, input_features):
        return [f"{col}_{c}" for col, cats in zip(input_features, self.categories_) for c in cats]
