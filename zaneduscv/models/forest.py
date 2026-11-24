"""Random Forest ve Extra Trees.

Random Forest: her agac bootstrap ornekle egitilir, her bolmede rastgele ozellik alt kumesi.
Extra Trees: bootstrap yok, esik degerleri de rastgele secilir (daha hizli, daha cesitli).
"""

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array
from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class _ForestBase(BaseEstimator):
    _splitter = "best"
    _bootstrap = True

    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1,
                 max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _fit_forest(self, X, y, tree_cls):
        X = check_array(X)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n) if self._bootstrap else np.arange(n)
            tree = tree_cls(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                            max_features=self.max_features, splitter=self._splitter)
            tree.fit(X[idx], y[idx], rng=rng)
            self.estimators_.append(tree)
        imp = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self


class RandomForestClassifier(_ForestBase, ClassifierMixin):
    def fit(self, X, y):
        y = np.asarray(y).astype(int).ravel()
        self.n_classes_ = max(int(y.max()) + 1, 2)
        return self._fit_forest(X, y, DecisionTreeClassifier)

    def predict_proba(self, X):
        X = check_array(X)
        total = np.zeros((len(X), self.n_classes_))
        for tree in self.estimators_:
            p = tree.predict_proba(X)
            total[:, :p.shape[1]] += p
        return total / len(self.estimators_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class RandomForestRegressor(_ForestBase, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1,
                 max_features=0.5, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_leaf, max_features, random_state)

    def fit(self, X, y):
        return self._fit_forest(X, np.asarray(y, dtype=float).ravel(), DecisionTreeRegressor)

    def predict(self, X):
        X = check_array(X)
        return np.mean([t.predict(X) for t in self.estimators_], axis=0)


class ExtraTreesClassifier(RandomForestClassifier):
    _splitter = "random"
    _bootstrap = False


class ExtraTreesRegressor(RandomForestRegressor):
    _splitter = "random"
    _bootstrap = False
