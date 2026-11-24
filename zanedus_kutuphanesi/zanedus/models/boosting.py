"""ZBoost: XGBoost mantigiyla yazdigimiz gradient boosting.

Her turda kaybin gradyani (g) ve hessian'i (h) hesaplaniyor, agac bu ikisine gore
kuruluyor: bolme kazanci G^2/(H+lambda), yaprak degeri -G/(H+lambda).
Destekler: learning_rate, max_depth, reg_lambda, gamma, subsample, colsample,
erken durdurma (early_stopping_rounds + validation_fraction).
"""

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array, sigmoid, softmax
from .tree import NewtonTree


class _ZBoostBase(BaseEstimator):
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=4, reg_lambda=1.0,
                 gamma=0.0, min_samples_leaf=1, subsample=0.8, colsample=0.8,
                 early_stopping_rounds=20, validation_fraction=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample = colsample
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    # alt siniflar: _init_score, _grad_hess, _loss, n_outputs
    def _boost(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        use_es = self.early_stopping_rounds and n >= 50 and self.validation_fraction > 0
        if use_es:
            perm = rng.permutation(n)
            n_val = max(1, int(n * self.validation_fraction))
            val, tr = perm[:n_val], perm[n_val:]
        else:
            tr, val = np.arange(n), np.array([], dtype=int)

        Xtr, ytr = X[tr], y[tr]
        self.base_score_ = self._init_score(ytr)
        F = np.tile(self.base_score_, (len(tr), 1))
        Fv = np.tile(self.base_score_, (len(val), 1))
        self.estimators_ = []
        best_loss, best_iter, bad = np.inf, 0, 0
        importances = np.zeros(X.shape[1])

        for it in range(self.n_estimators):
            G, H = self._grad_hess(ytr, F)
            rows = rng.random(len(tr)) < self.subsample if self.subsample < 1 else np.ones(len(tr), bool)
            if rows.sum() < 2:
                rows[:] = True
            round_trees = []
            for k in range(self.n_outputs_):
                tree = NewtonTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                  max_features=self.colsample if self.colsample < 1 else None,
                                  reg_lambda=self.reg_lambda, gamma=self.gamma)
                tree.fit(Xtr[rows], G[rows, k], H[rows, k], rng=rng)
                F[:, k] += self.learning_rate * tree.predict(Xtr)
                if len(val):
                    Fv[:, k] += self.learning_rate * tree.predict(X[val])
                importances += tree.tree_[5]
                round_trees.append(tree)
            self.estimators_.append(round_trees)

            if len(val):
                loss = self._loss(y[val], Fv)
                if loss < best_loss - 1e-9:
                    best_loss, best_iter, bad = loss, it + 1, 0
                else:
                    bad += 1
                    if bad >= self.early_stopping_rounds:
                        break
        if len(val):
            self.estimators_ = self.estimators_[:max(best_iter, 1)]
        self.best_iteration_ = len(self.estimators_)
        self.feature_importances_ = importances / importances.sum() if importances.sum() > 0 else importances
        return self

    def _raw(self, X):
        X = check_array(X)
        F = np.tile(self.base_score_, (len(X), 1))
        for round_trees in self.estimators_:
            for k, tree in enumerate(round_trees):
                F[:, k] += self.learning_rate * tree.predict(X)
        return F


class ZBoostClassifier(_ZBoostBase, ClassifierMixin):
    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y).astype(int).ravel()
        self.n_classes_ = max(int(y.max()) + 1, 2)
        self.n_outputs_ = 1 if self.n_classes_ == 2 else self.n_classes_
        return self._boost(X, y)

    def _init_score(self, y):
        if self.n_outputs_ == 1:
            p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            return np.array([np.log(p / (1 - p))])
        freq = np.bincount(y, minlength=self.n_classes_) / len(y)
        return np.log(np.clip(freq, 1e-6, None))

    def _grad_hess(self, y, F):
        if self.n_outputs_ == 1:
            p = sigmoid(F[:, 0])
            return (p - y)[:, None], np.maximum(p * (1 - p), 1e-6)[:, None]
        P = softmax(F)
        Y = np.eye(self.n_classes_)[y]
        return P - Y, np.maximum(P * (1 - P), 1e-6)

    def _loss(self, y, F):
        P = self._to_proba(F)
        return -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None)))

    def _to_proba(self, F):
        if self.n_outputs_ == 1:
            p = sigmoid(F[:, 0])
            return np.column_stack([1 - p, p])
        return softmax(F)

    def predict_proba(self, X):
        return self._to_proba(self._raw(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class ZBoostRegressor(_ZBoostBase, RegressorMixin):
    n_outputs_ = 1

    def fit(self, X, y):
        return self._boost(check_array(X), np.asarray(y, dtype=float).ravel())

    def _init_score(self, y):
        return np.array([y.mean()])

    def _grad_hess(self, y, F):
        return (F[:, 0] - y)[:, None], np.ones((len(y), 1))

    def _loss(self, y, F):
        return float(np.mean((F[:, 0] - y) ** 2))

    def predict(self, X):
        return self._raw(X)[:, 0]
