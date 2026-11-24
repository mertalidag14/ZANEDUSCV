"""ZANEDUSCV modelleri: ZANEDUSCVClassifier ve ZANEDUSCVRegressor."""

import pickle

import numpy as np
import pandas as pd

from ._booster import Booster, LogLoss, SoftmaxLoss, SquaredLoss
from ._data import DataProcessor
from .base import BaseEstimator, ClassifierMixin, RegressorMixin, sigmoid, softmax


class _ZANEDUSCVBase(BaseEstimator):
    def __init__(self, n_estimators=500, learning_rate=0.05, max_depth=5, min_samples_leaf=5,
                 min_child_weight=1e-3, reg_lambda=1.0, gamma=0.0, subsample=0.8, colsample=0.8,
                 max_bins=64, oblique=True, oblique_top=3, linear_leaves=True, linear_lambda=1.0,
                 min_samples_linear=30, confidence_kappa=10.0, early_stopping_rounds=50,
                 validation_fraction=0.1, random_state=None, verbose=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
        self.max_bins = max_bins
        self.oblique = oblique
        self.oblique_top = oblique_top
        self.linear_leaves = linear_leaves
        self.linear_lambda = linear_lambda
        self.min_samples_linear = min_samples_linear
        self.confidence_kappa = confidence_kappa
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ ortak
    def _encode_y(self, y):
        raise NotImplementedError

    def _make_loss(self):
        raise NotImplementedError

    def _split_val(self, n, y_enc, rng):
        idx = rng.permutation(n)
        n_val = max(1, int(round(n * self.validation_fraction)))
        if self.task == "classification":         # her siniftan orantili ayir
            val = []
            for c in np.unique(y_enc):
                ci = idx[y_enc[idx] == c]
                val.extend(ci[: max(1, int(round(len(ci) * self.validation_fraction)))])
            val = np.array(sorted(val))
        else:
            val = np.sort(idx[:n_val])
        mask = np.zeros(n, dtype=bool)
        mask[val] = True
        return np.where(~mask)[0], val

    def fit(self, X, y, cat_features=None, eval_set=None):
        rng = np.random.default_rng(self.random_state)
        X = self._frame(X)
        y_enc = self._encode_y(np.asarray(y).ravel())
        if len(X) != len(y_enc):
            raise ValueError("X ve y ayni sayida satir icermeli")
        self.processor_ = DataProcessor(max_bins=self.max_bins, cat_features=cat_features)

        if eval_set is not None:
            Xv, yv = eval_set
            Xv = self._frame(Xv, X.columns)
            yv = self._transform_y(np.asarray(yv).ravel())
            tr = np.arange(len(X))
        elif self.early_stopping_rounds and self.validation_fraction and len(X) >= 50:
            tr, va = self._split_val(len(X), y_enc, rng)
            Xv, yv = X.iloc[va], y_enc[va]
        else:
            tr, Xv, yv = np.arange(len(X)), None, None

        self.processor_.fit(X.iloc[tr])
        B, Z = self.processor_.transform(X.iloc[tr])
        Bv = Zv = None
        if Xv is not None:
            Bv, Zv = self.processor_.transform(Xv)
        self.feature_names_in_ = self.processor_.feature_names_
        self.n_features_in_ = len(self.feature_names_in_)

        params = {k: getattr(self, k) for k in self._param_names()}
        if Xv is None:
            params["early_stopping_rounds"] = 0
        self.booster_ = Booster(self._make_loss(), params, rng)
        self.booster_.fit(B, Z, y_enc[tr], self.processor_, Bv, Zv, yv, verbose=self.verbose)
        self.best_iteration_ = self.booster_.best_iteration_
        self.feature_importances_ = self.booster_.feature_importances_
        return self

    def _raw(self, X):
        X = self._frame(X, self.feature_names_in_)
        B, Z = self.processor_.transform(X)
        return self.booster_.raw_predict(B, Z)

    @staticmethod
    def _frame(X, columns=None):
        if isinstance(X, pd.DataFrame):
            df = X.reset_index(drop=True)
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            df = pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])
            if arr.dtype != object:
                df = df.astype(float)
        if columns is not None and isinstance(X, pd.DataFrame):
            missing = [c for c in columns if str(c) not in map(str, df.columns)]
            if missing:
                raise ValueError(f"Su kolonlar eksik: {missing}")
            df.columns = [str(c) for c in df.columns]
            df = df[[str(c) for c in columns]]
        return df

    def get_feature_importance(self):
        """Kolon adi -> onem (toplam 1) seklinde siralanmis pandas Series."""
        return pd.Series(self.feature_importances_, index=self.feature_names_in_) \
            .sort_values(ascending=False)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class ZANEDUSCVClassifier(_ZANEDUSCVBase, ClassifierMixin):
    """Siniflandirma icin ZANEDUSCV. Etiketler yazi, sayi ya da bool olabilir."""

    def _encode_y(self, y):
        self.classes_, enc = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("En az 2 sinif olmali")
        return enc

    def _transform_y(self, y):
        pos = {c: i for i, c in enumerate(self.classes_)}
        return np.array([pos[v] for v in y])

    def _make_loss(self):
        return LogLoss() if len(self.classes_) == 2 else SoftmaxLoss(len(self.classes_))

    def predict_proba(self, X):
        F = self._raw(X)
        if len(self.classes_) == 2:
            p = sigmoid(F[:, 0])
            return np.column_stack([1 - p, p])
        return softmax(F)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ZANEDUSCVRegressor(_ZANEDUSCVBase, RegressorMixin):
    """Regresyon icin ZANEDUSCV."""

    def _encode_y(self, y):
        return np.asarray(y, dtype=float)

    def _transform_y(self, y):
        return np.asarray(y, dtype=float)

    def _make_loss(self):
        return SquaredLoss()

    def predict(self, X):
        return self._raw(X)[:, 0]
