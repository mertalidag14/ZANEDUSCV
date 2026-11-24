"""Lojistik regresyon ve Ridge regresyon, gradyan inisi ile."""

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_array, softmax


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Cok sinifli (softmax) lojistik regresyon.

    Kayip: cross-entropy + L2 ceza. Optimizasyon: Adam ile tam batch gradyan inisi.
    Etiketler 0..k-1 olmali (AutoML bunu kendisi ayarliyor).
    """

    def __init__(self, C=1.0, lr=0.1, max_iter=500, tol=1e-6, random_state=None):
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y).astype(int).ravel()
        n, d = X.shape
        self.classes_ = np.unique(y)
        k = max(int(y.max()) + 1, 2)
        Y = np.eye(k)[y]
        W = np.zeros((d, k))
        b = np.zeros(k)
        lam = 1.0 / (self.C * n)
        m = [np.zeros_like(W), np.zeros_like(b)]
        v = [np.zeros_like(W), np.zeros_like(b)]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        prev = np.inf
        for t in range(1, self.max_iter + 1):
            P = softmax(X @ W + b)
            loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1)) + 0.5 * lam * np.sum(W ** 2)
            G = (P - Y) / n
            grads = [X.T @ G + lam * W, G.sum(axis=0)]
            for i, (param, g) in enumerate(zip((W, b), grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        self.coef_, self.intercept_, self.n_iter_ = W, b, t
        return self

    def predict_proba(self, X):
        return softmax(check_array(X) @ self.coef_ + self.intercept_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class RidgeRegression(BaseEstimator, RegressorMixin):
    """L2 cezali dogrusal regresyon. Kapali form cozum: w = (X'X + aI)^-1 X'y"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y, dtype=float).ravel()
        self.x_mean_, self.y_mean_ = X.mean(axis=0), y.mean()
        Xc, yc = X - self.x_mean_, y - self.y_mean_
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_
        return self

    def predict(self, X):
        return check_array(X) @ self.coef_ + self.intercept_
