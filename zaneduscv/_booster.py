"""Boosting dongusu: agaclari art arda ekleyerek kaybi Newton adimlariyla dusuruyor."""

import numpy as np

from ._tree import Tree
from .base import sigmoid, softmax


class SquaredLoss:
    n_outputs = 1

    def init_score(self, y):
        return np.array([y.mean()])

    def grad_hess(self, y, F):
        return (F[:, 0] - y)[:, None], np.ones((len(y), 1))

    def loss(self, y, F):
        return float(np.mean((F[:, 0] - y) ** 2))


class LogLoss:
    n_outputs = 1

    def init_score(self, y):
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        return np.array([np.log(p / (1 - p))])

    def grad_hess(self, y, F):
        p = sigmoid(F[:, 0])
        return (p - y)[:, None], np.maximum(p * (1 - p), 1e-6)[:, None]

    def loss(self, y, F):
        p = np.clip(sigmoid(F[:, 0]), 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class SoftmaxLoss:
    def __init__(self, k):
        self.n_outputs = k

    def init_score(self, y):
        p = np.bincount(y, minlength=self.n_outputs) / len(y)
        return np.log(np.clip(p, 1e-6, None))

    def grad_hess(self, y, F):
        P = softmax(F)
        Y = np.zeros_like(P)
        Y[np.arange(len(y)), y] = 1.0
        return P - Y, np.maximum(2.0 * P * (1 - P), 1e-6)

    def loss(self, y, F):
        P = np.clip(softmax(F), 1e-15, None)
        return float(-np.mean(np.log(P[np.arange(len(y)), y])))


class Booster:
    def __init__(self, loss, params, rng):
        self.loss = loss
        self.p = params
        self.rng = rng

    def _tree(self):
        p = self.p
        return Tree(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                    min_child_weight=p["min_child_weight"], reg_lambda=p["reg_lambda"],
                    gamma=p["gamma"], oblique=p["oblique"], oblique_top=p["oblique_top"],
                    linear_leaves=p["linear_leaves"], linear_lambda=p["linear_lambda"],
                    min_samples_linear=p["min_samples_linear"],
                    confidence_kappa=p["confidence_kappa"])

    def fit(self, B, Z, y, proc, B_val=None, Z_val=None, y_val=None, verbose=0):
        p, loss, rng = self.p, self.loss, self.rng
        n, d = B.shape
        K = loss.n_outputs
        self.base_ = loss.init_score(y)
        F = np.tile(self.base_, (n, 1))
        has_val = B_val is not None
        if has_val:
            Fv = np.tile(self.base_, (len(B_val), 1))
        self.trees_ = []
        self.val_curve_ = []
        best, best_iter, wait = np.inf, 0, 0
        lr = p["learning_rate"]
        all_feats = np.arange(d)

        for it in range(p["n_estimators"]):
            g, h = loss.grad_hess(y, F)
            if p["subsample"] < 1.0:
                rows = rng.choice(n, max(2, int(n * p["subsample"])), replace=False)
            else:
                rows = None
            if p["colsample"] < 1.0 and d > 1:
                feats = np.sort(rng.choice(d, max(1, int(round(d * p["colsample"]))), replace=False))
            else:
                feats = all_feats
            round_trees = []
            for k in range(K):
                t = self._tree()
                if rows is None:
                    t.fit(B, Z, g[:, k], h[:, k], proc.is_cat_, proc.n_bins_, feats, proc.total_bins_)
                else:
                    t.fit(B[rows], Z[rows], g[rows, k], h[rows, k], proc.is_cat_, proc.n_bins_,
                          feats, proc.total_bins_)
                F[:, k] += lr * t.predict(B, Z)
                if has_val:
                    Fv[:, k] += lr * t.predict(B_val, Z_val)
                round_trees.append(t)
            self.trees_.append(round_trees)

            if has_val:
                v = loss.loss(y_val, Fv)
                self.val_curve_.append(v)
                if v < best - 1e-7:
                    best, best_iter, wait = v, it + 1, 0
                else:
                    wait += 1
                if verbose and (it + 1) % verbose == 0:
                    print(f"[{it + 1}] dogrulama kaybi = {v:.5f}")
                if p["early_stopping_rounds"] and wait >= p["early_stopping_rounds"]:
                    break
            elif verbose and (it + 1) % verbose == 0:
                print(f"[{it + 1}] egitim kaybi = {loss.loss(y, F):.5f}")

        if has_val and best_iter > 0:
            self.trees_ = self.trees_[:best_iter]
            self.best_iteration_ = best_iter
        else:
            self.best_iteration_ = len(self.trees_)
        imp = np.zeros(d)
        for rt in self.trees_:
            for t in rt:
                imp += t.gain_by_feature
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    def raw_predict(self, B, Z, n_iter=None):
        F = np.tile(self.base_, (len(B), 1))
        lr = self.p["learning_rate"]
        for rt in self.trees_[:n_iter]:
            for k, t in enumerate(rt):
                F[:, k] += lr * t.predict(B, Z)
        return F
