"""ZANEDUSCV'nin kalbi: AutoML sinifi.

Kullanim:
    from zaneduscv import AutoML
    automl = AutoML()
    automl.fit(X_train, y_train)
    automl.leaderboard()
    automl.predict(X_test)
"""

import time

import numpy as np

from .base import BaseEstimator, clone
from .metrics import get_metric
from .model_selection import KFold, StratifiedKFold, cross_val_score, score_estimator
from .models import get_models
from .pipeline import Pipeline
from .preprocessing import AutoPreprocessor, LabelEncoder
from .reporting import make_leaderboard
from .utils import detect_task, to_dataframe

DEFAULT_METRICS = {"classification": "accuracy", "regression": "r2"}


class AutoML(BaseEstimator):
    """Veriyi on isler, model havuzundaki her modeli capraz dogrulama ile dener
    ve en iyisini tum veriyle yeniden egitir.

    Parametreler
    ------------
    task : "auto", "classification" ya da "regression"
    models : denenecek model isimleri, None ise hepsi
    metric : "accuracy", "f1_macro", "roc_auc", "log_loss", "r2", "rmse", "mae" ...
    cv : kac katli capraz dogrulama
    preprocessing : AutoPreprocessor ayarlari (dict)
    random_state : tekrarlanabilirlik icin seed
    verbose : True ise her modelin skorunu ekrana yazar
    """

    def __init__(self, task="auto", models=None, metric=None, cv=5,
                 preprocessing=None, random_state=42, verbose=True):
        self.task = task
        self.models = models
        self.metric = metric
        self.cv = cv
        self.preprocessing = preprocessing
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _make_pipeline(self, model):
        return Pipeline([("preprocessor", AutoPreprocessor(**(self.preprocessing or {}))),
                         ("model", model)])

    def fit(self, X, y):
        X = to_dataframe(X)
        y = np.asarray(y).ravel()
        if len(X) != len(y):
            raise ValueError("X ve y ayni sayida satira sahip olmali")

        self.task_ = detect_task(y) if self.task == "auto" else self.task
        self.metric_ = self.metric or DEFAULT_METRICS[self.task_]
        _, self.greater_is_better_, _ = get_metric(self.metric_)

        if self.task_ == "classification":
            self.label_encoder_ = LabelEncoder().fit(y)
            y_fit = self.label_encoder_.transform(y)
            self.classes_ = self.label_encoder_.classes_
            splitter = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        else:
            self.label_encoder_ = None
            y_fit = y.astype(float)
            splitter = KFold(self.cv, shuffle=True, random_state=self.random_state)

        candidates = get_models(self.models, self.task_, self.random_state)
        self._log(f"[ZANEDUSCV] gorev: {self.task_} | metrik: {self.metric_} | "
                  f"{len(candidates)} model deneniyor")

        self.results_ = {}
        for name, model in candidates.items():
            start = time.perf_counter()
            try:
                with np.errstate(all="ignore"):
                    scores = cross_val_score(self._make_pipeline(model), X, y_fit,
                                             cv=splitter, metric=self.metric_)
                res = {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "status": "ok"}
            except Exception as exc:  # bir model patlarsa digerleri devam etsin
                res = {"mean": np.nan, "std": np.nan, "status": f"hata: {exc}"}
            res["fit_time"] = round(time.perf_counter() - start, 2)
            self.results_[name] = res
            self._log(f"  {name:<14} {res['mean']:.4f} (+/- {res['std']:.4f})  {res['fit_time']}s")

        ok = {n: r for n, r in self.results_.items() if r["status"] == "ok"}
        if not ok:
            raise RuntimeError("Hicbir model egitilemedi, results_ icindeki hatalara bak")

        sign = 1 if self.greater_is_better_ else -1
        self.best_model_name_ = max(ok, key=lambda n: sign * ok[n]["mean"])
        self.best_score_ = ok[self.best_model_name_]["mean"]
        self.best_pipeline_ = self._make_pipeline(clone(candidates[self.best_model_name_]))
        self.best_pipeline_.fit(X, y_fit)
        self.feature_names_in_ = list(X.columns)
        self._log(f"[ZANEDUSCV] en iyi model: {self.best_model_name_} ({self.best_score_:.4f})")
        return self

    def _check_fitted(self):
        if not hasattr(self, "best_pipeline_"):
            raise RuntimeError("Once fit() cagirilmali")

    def predict(self, X):
        self._check_fitted()
        pred = self.best_pipeline_.predict(to_dataframe(X))
        if self.label_encoder_ is not None:
            pred = self.label_encoder_.inverse_transform(np.asarray(pred).astype(int).ravel())
        return pred

    def predict_proba(self, X):
        self._check_fitted()
        if self.task_ != "classification":
            raise AttributeError("predict_proba sadece siniflandirmada var")
        return self.best_pipeline_.predict_proba(to_dataframe(X))

    def score(self, X, y):
        """Secilen metrikle skor dondurur."""
        self._check_fitted()
        y = np.asarray(y).ravel()
        if self.label_encoder_ is not None:
            y = self.label_encoder_.transform(y)
        return score_estimator(self.best_pipeline_, to_dataframe(X), y, self.metric_)

    def leaderboard(self):
        """Tum modellerin siralamasini DataFrame olarak dondurur."""
        if not hasattr(self, "results_"):
            raise RuntimeError("Once fit() cagirilmali")
        return make_leaderboard(self.results_, self.metric_, self.greater_is_better_)

    def feature_importances(self):
        """En iyi model agac tabanliysa ozellik onemlerini dondurur."""
        self._check_fitted()
        model = self.best_pipeline_.final_estimator_
        if not hasattr(model, "feature_importances_"):
            raise AttributeError(f"{self.best_model_name_} ozellik onemi vermiyor")
        import pandas as pd
        names = self.best_pipeline_.fitted_steps_[0][1].get_feature_names_out()
        return pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
