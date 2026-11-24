"""Karar agaclari (CART), sifirdan.

Tek bir agac motoru uc is icin kullaniliyor:
  - siniflandirma (Gini)
  - regresyon (MSE)
  - gradient boosting (XGBoost tarzi gradyan/hessian kazanci)

Fikir: her dugumde her ozellik icin veriyi siraliyoruz, kumulatif toplamlarla
tum olasi bolmelerin kazancini tek seferde hesapliyoruz. "Kalite" fonksiyonu
bir istatistik toplamindan skor uretiyor; kazanc = sol + sag - ebeveyn.
"""

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array


# ----------------------------------------------------------------------------
# Kriterler: istatistik matrisi (n, s) ve bu istatistiklerin toplamindan skor
# ----------------------------------------------------------------------------

class _Gini:
    """stats = one-hot etiketler. skor = sum(c_k^2) / n  (buyuk olan daha saf)."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def stats(self, y):
        return np.eye(self.n_classes)[y.astype(int)]

    @staticmethod
    def count(S):
        return S.sum(axis=-1)

    def score(self, S):
        n = S.sum(axis=-1)
        return np.divide((S ** 2).sum(axis=-1), n, out=np.zeros_like(n), where=n > 0)

    @staticmethod
    def leaf(S):
        return S / S.sum()


class _MSE:
    """stats = [y, 1]. skor = (sum y)^2 / n  (varyansi en cok azaltan bolme)."""

    def stats(self, y):
        return np.column_stack([y, np.ones_like(y)])

    @staticmethod
    def count(S):
        return S[..., 1]

    def score(self, S):
        n = S[..., 1]
        return np.divide(S[..., 0] ** 2, n, out=np.zeros_like(n), where=n > 0)

    @staticmethod
    def leaf(S):
        return np.array([S[0] / S[1]])


class _Newton:
    """XGBoost kazanci. stats = [g, h, 1]. skor = G^2 / (H + lambda)."""

    def __init__(self, reg_lambda=1.0):
        self.reg_lambda = reg_lambda

    @staticmethod
    def count(S):
        return S[..., 2]

    def score(self, S):
        return S[..., 0] ** 2 / (S[..., 1] + self.reg_lambda)

    def leaf(self, S):
        return np.array([-S[0] / (S[1] + self.reg_lambda)])


# ----------------------------------------------------------------------------
# Agac motoru
# ----------------------------------------------------------------------------

class _TreeBuilder:
    def __init__(self, criterion, max_depth, min_samples_split, min_samples_leaf,
                 max_features, splitter, min_gain, rng):
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.min_gain = min_gain
        self.rng = rng

    def build(self, X, stats):
        self.feature, self.threshold, self.left, self.right, self.value = [], [], [], [], []
        self.n_features = X.shape[1]
        self.importances = np.zeros(X.shape[1])
        self._grow(X, stats, np.arange(len(X)), depth=0)
        return (np.array(self.feature), np.array(self.threshold), np.array(self.left),
                np.array(self.right), np.array(self.value), self.importances)

    def _new_node(self, value):
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(value)
        return len(self.feature) - 1

    def _features_to_try(self):
        d = self.n_features
        k = self.max_features
        if k is None:
            k = d
        elif k == "sqrt":
            k = max(1, int(np.sqrt(d)))
        elif k == "log2":
            k = max(1, int(np.log2(d)))
        elif isinstance(k, float):
            k = max(1, int(k * d))
        k = min(int(k), d)
        if k == d:
            return np.arange(d)
        return self.rng.choice(d, size=k, replace=False)

    def _best_split(self, X, S, total):
        crit = self.criterion
        parent = crit.score(total)
        m = self.min_samples_leaf
        best = (self.min_gain, None, None)
        for j in self._features_to_try():
            x = X[:, j]
            if self.splitter == "random":
                lo, hi = x.min(), x.max()
                if lo == hi:
                    continue
                thr = self.rng.uniform(lo, hi)
                mask = x <= thr
                L = S[mask].sum(axis=0)
                nl = crit.count(L)
                if nl < m or crit.count(total) - nl < m:
                    continue
                gain = crit.score(L) + crit.score(total - L) - parent
                if gain > best[0]:
                    best = (gain, j, thr)
                continue

            order = np.argsort(x, kind="mergesort")
            xs = x[order]
            cum = np.cumsum(S[order], axis=0)[:-1]   # sol taraf: ilk i+1 ornek
            valid = xs[:-1] < xs[1:]                  # ayni degerin ortasindan bolme
            n_left = crit.count(cum)
            n_total = crit.count(total)
            valid &= (n_left >= m) & (n_total - n_left >= m)
            if not valid.any():
                continue
            gains = crit.score(cum) + crit.score(total - cum) - parent
            gains[~valid] = -np.inf
            i = int(np.argmax(gains))
            if gains[i] > best[0]:
                best = (gains[i], j, 0.5 * (xs[i] + xs[i + 1]))
        return best

    def _grow(self, X, S, idx, depth):
        Xn, Sn = X[idx], S[idx]
        total = Sn.sum(axis=0)
        node = self._new_node(self.criterion.leaf(total))
        if depth >= self.max_depth or len(idx) < self.min_samples_split:
            return node
        gain, j, thr = self._best_split(Xn, Sn, total)
        if j is None:
            return node
        mask = Xn[:, j] <= thr
        self.importances[j] += gain
        self.feature[node] = j
        self.threshold[node] = thr
        self.left[node] = self._grow(X, S, idx[mask], depth + 1)
        self.right[node] = self._grow(X, S, idx[~mask], depth + 1)
        return node


def _apply(tree, X):
    """Her ornegin dustugu yaprak indeksini dondurur (vektorel)."""
    feature, threshold, left, right = tree[:4]
    nodes = np.zeros(len(X), dtype=int)
    active = feature[nodes] >= 0
    while active.any():
        a = np.where(active)[0]
        f = feature[nodes[a]]
        go_left = X[a, f] <= threshold[nodes[a]]
        nodes[a] = np.where(go_left, left[nodes[a]], right[nodes[a]])
        active[a] = feature[nodes[a]] >= 0
    return nodes


class _BaseTree(BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, splitter="best", random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.random_state = random_state

    def _fit_tree(self, X, stats, criterion, rng=None, min_gain=1e-12):
        rng = rng if rng is not None else np.random.default_rng(self.random_state)
        builder = _TreeBuilder(criterion, self.max_depth, self.min_samples_split,
                               self.min_samples_leaf, self.max_features, self.splitter,
                               min_gain, rng)
        self.tree_ = builder.build(X, stats)
        imp = self.tree_[5]
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        self.n_features_in_ = X.shape[1]
        return self

    def _leaf_values(self, X):
        return self.tree_[4][_apply(self.tree_, check_array(X))]


class DecisionTreeClassifier(_BaseTree, ClassifierMixin):
    def fit(self, X, y, rng=None):
        X = check_array(X)
        y = np.asarray(y).astype(int).ravel()
        self.n_classes_ = max(int(y.max()) + 1, 2)
        crit = _Gini(self.n_classes_)
        return self._fit_tree(X, crit.stats(y), crit, rng)

    def predict_proba(self, X):
        return self._leaf_values(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class DecisionTreeRegressor(_BaseTree, RegressorMixin):
    def fit(self, X, y, rng=None):
        X = check_array(X)
        y = np.asarray(y, dtype=float).ravel()
        crit = _MSE()
        return self._fit_tree(X, crit.stats(y), crit, rng)

    def predict(self, X):
        return self._leaf_values(X)[:, 0]


class NewtonTree(_BaseTree):
    """Boosting icin: gradyan ve hessian'dan agac kurar (XGBoost mantigi)."""

    def __init__(self, max_depth=4, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, reg_lambda=1.0, gamma=0.0, random_state=None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features,
                         "best", random_state)
        self.reg_lambda = reg_lambda
        self.gamma = gamma

    def fit(self, X, g, h, rng=None):
        stats = np.column_stack([g, h, np.ones_like(g)])
        # gamma: bolme icin gereken en az kazanc (XGBoost'taki gibi)
        return self._fit_tree(X, stats, _Newton(self.reg_lambda), rng,
                              min_gain=max(self.gamma, 1e-12))

    def predict(self, X):
        return self._leaf_values(X)[:, 0]
