"""Adimlari arka arkaya baglayan basit Pipeline."""

from .base import BaseEstimator, clone


class Pipeline(BaseEstimator):
    """steps: [(isim, donusturucu), ..., (isim, model)]"""

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        self.fitted_steps_ = []
        for name, step in self.steps[:-1]:
            step = clone(step)
            X = step.fit_transform(X, y)
            self.fitted_steps_.append((name, step))
        name, model = self.steps[-1]
        model = clone(model)
        model.fit(X, y)
        self.fitted_steps_.append((name, model))
        return self

    def _transform(self, X):
        for _, step in self.fitted_steps_[:-1]:
            X = step.transform(X)
        return X

    @property
    def final_estimator_(self):
        return self.fitted_steps_[-1][1]

    @property
    def task(self):
        return getattr(self.steps[-1][1], "task", None)

    def predict(self, X):
        return self.final_estimator_.predict(self._transform(X))

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(self._transform(X))
