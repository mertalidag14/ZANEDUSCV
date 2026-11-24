"""Tum modellerin ve donusturuculerin ortak taban siniflari.

scikit-learn kullanmadan, onun API'sine benzer bir yapi kuruyoruz:
fit / predict / transform / get_params / set_params / clone.
"""

import copy
import inspect

import numpy as np


class BaseEstimator:
    """get_params, set_params ve okunabilir repr saglar."""

    @classmethod
    def _param_names(cls):
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self._param_names():
                raise ValueError(f"{type(self).__name__} icin gecersiz parametre: {key}")
            setattr(self, key, value)
        return self

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


def _sklearn_tags(estimator_type):
    """scikit-learn 1.6+ araclari (GridSearchCV, cross_val_score ...) icin etiketler.

    ZANEDUSCV sklearn'e ihtiyac duymuyor; sklearn yuklu degilse bu hic kullanilmiyor.
    """
    try:
        from sklearn.utils import (ClassifierTags, InputTags, RegressorTags, Tags,
                                   TargetTags)
    except ImportError as exc:
        raise AttributeError("__sklearn_tags__") from exc
    return Tags(
        estimator_type=estimator_type,
        target_tags=TargetTags(required=True),
        classifier_tags=ClassifierTags() if estimator_type == "classifier" else None,
        regressor_tags=RegressorTags() if estimator_type == "regressor" else None,
        input_tags=InputTags(allow_nan=True, categorical=True, string=True),
    )


class ClassifierMixin:
    task = "classification"
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        return _sklearn_tags("classifier")

    def score(self, X, y):
        from .metrics import accuracy
        return accuracy(y, self.predict(X))


class RegressorMixin:
    task = "regression"
    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        return _sklearn_tags("regressor")

    def score(self, X, y):
        from .metrics import r2
        return r2(y, self.predict(X))


class TransformerMixin:
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def clone(estimator):
    """Ayni parametrelerle egitilmemis yeni bir kopya olusturur."""
    params = {k: copy.deepcopy(v) for k, v in estimator.get_params().items()}
    return type(estimator)(**params)


def check_array(X):
    """Girdiyi 2 boyutlu float numpy dizisine cevirir."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
