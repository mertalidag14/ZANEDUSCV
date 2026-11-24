"""Destek Vektor Makineleri.

kernel="linear": dogrudan dogrusal SVM.
kernel="rbf": RBF cekirdegini Random Fourier Features ile yaklasik hesaplayip
uzerine dogrusal SVM kuruyoruz (Rahimi & Recht, 2007). Boylece buyuk veride de hizli.
Kayip: squared hinge (siniflandirma) / epsilon-insensitive squared (regresyon), L2 ceza.
"""

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array, softmax


class _RFF:
    def __init__(self, n_components, gamma, d, rng):
        self.W = rng.normal(scale=np.sqrt(2 * gamma), size=(d, n_components))
        self.b = rng.uniform(0, 2 * np.pi, size=n_components)
        self.scale = np.sqrt(2.0 / n_components)

    def __call__(self, X):
        return self.scale * np.cos(X @ self.W + self.b)


class _SVMBase(BaseEstimator):
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", n_components=300,
                 lr=0.05, max_iter=300, random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state

    def _features(self, X, fit=False):
        X = check_array(X)
        if self.kernel == "linear":
            return X
        if self.kernel != "rbf":
            raise ValueError("kernel 'linear' ya da 'rbf' olmali")
        if fit:
            gamma = self.gamma
            if gamma == "scale":
                var = X.var()
                gamma = 1.0 / (X.shape[1] * var) if var > 0 else 1.0
            rng = np.random.default_rng(self.random_state)
            self.rff_ = _RFF(self.n_components, gamma, X.shape[1], rng)
        return self.rff_(X)

    def _optimize(self, Z, grad_fn, k):
        """Adam ile tam batch optimizasyon. grad_fn(scores) -> dL/dscores"""
        n, d = Z.shape
        W, b = np.zeros((d, k)), np.zeros(k)
        lam = 1.0 / (self.C * n)
        m = [np.zeros_like(W), np.zeros_like(b)]
        v = [np.zeros_like(W), np.zeros_like(b)]
        for t in range(1, self.max_iter + 1):
            G = grad_fn(Z @ W + b) / n
            grads = [Z.T @ G + lam * W, G.sum(axis=0)]
            for i, (param, g) in enumerate(zip((W, b), grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g ** 2
                param -= self.lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        return W, b


class SVMClassifier(_SVMBase, ClassifierMixin):
    """Bire-karsi-hepsi (one-vs-rest) SVM."""

    def fit(self, X, y):
        y = np.asarray(y).astype(int).ravel()
        Z = self._features(X, fit=True)
        k = max(int(y.max()) + 1, 2)
        T = np.where(np.eye(k)[y] == 1, 1.0, -1.0)  # her sinif icin +1 / -1

        def grad(scores):
            margin = np.maximum(0, 1 - T * scores)
            return -2 * T * margin

        self.coef_, self.intercept_ = self._optimize(Z, grad, k)
        self.classes_ = np.arange(k)
        return self

    def decision_function(self, X):
        return self._features(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        # kalibrasyonsuz, skorlarin softmax'i (siralama icin yeterli)
        return softmax(2 * self.decision_function(X))

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)


class SVMRegressor(_SVMBase, RegressorMixin):
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", n_components=300,
                 lr=0.05, max_iter=500, epsilon=0.1, random_state=None):
        super().__init__(C, kernel, gamma, n_components, lr, max_iter, random_state)
        self.epsilon = epsilon

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        self.y_mean_, self.y_std_ = y.mean(), y.std() or 1.0
        t = ((y - self.y_mean_) / self.y_std_)[:, None]
        Z = self._features(X, fit=True)
        eps = self.epsilon

        def grad(scores):
            r = scores - t
            return 2 * np.sign(r) * np.maximum(0, np.abs(r) - eps)

        self.coef_, self.intercept_ = self._optimize(Z, grad, 1)
        return self

    def predict(self, X):
        out = (self._features(X) @ self.coef_ + self.intercept_).ravel()
        return out * self.y_std_ + self.y_mean_
