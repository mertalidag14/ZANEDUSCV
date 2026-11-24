"""Siniflandirma ve regresyon icin model havuzu.

XGBoost, LightGBM ve CatBoost kurulu degilse sessizce atlanir,
boylece kutuphane sadece scikit-learn ile de calisir.
"""

from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR


def _xgb(task, seed):
    import xgboost as xgb
    if task == "classification":
        return xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=seed,
                                 n_jobs=-1, verbosity=0)
    return xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=seed,
                            n_jobs=-1, verbosity=0)


def _lgbm(task, seed):
    import lightgbm as lgb
    if task == "classification":
        return lgb.LGBMClassifier(n_estimators=300, random_state=seed, verbose=-1)
    return lgb.LGBMRegressor(n_estimators=300, random_state=seed, verbose=-1)


def _catboost(task, seed):
    import catboost as cb
    if task == "classification":
        return cb.CatBoostClassifier(iterations=300, random_seed=seed, verbose=0)
    return cb.CatBoostRegressor(iterations=300, random_seed=seed, verbose=0)


_BUILDERS = {
    "logreg": {
        "classification": lambda s: LogisticRegression(max_iter=2000, random_state=s),
        "regression": lambda s: Ridge(random_state=s),
    },
    "knn": {
        "classification": lambda s: KNeighborsClassifier(),
        "regression": lambda s: KNeighborsRegressor(),
    },
    "svm": {
        "classification": lambda s: SVC(probability=True, random_state=s),
        "regression": lambda s: SVR(),
    },
    "random_forest": {
        "classification": lambda s: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=s),
        "regression": lambda s: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=s),
    },
    "extra_trees": {
        "classification": lambda s: ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=s),
        "regression": lambda s: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=s),
    },
    "xgboost": {"classification": lambda s: _xgb("classification", s),
                "regression": lambda s: _xgb("regression", s)},
    "lightgbm": {"classification": lambda s: _lgbm("classification", s),
                 "regression": lambda s: _lgbm("regression", s)},
    "catboost": {"classification": lambda s: _catboost("classification", s),
                 "regression": lambda s: _catboost("regression", s)},
}

MODEL_NAMES = list(_BUILDERS)


def get_model(name, task="classification", random_state=42):
    """Tek bir modeli ismiyle olusturur."""
    if name not in _BUILDERS:
        raise ValueError(f"Bilinmeyen model: {name}. Secenekler: {MODEL_NAMES}")
    if task not in ("classification", "regression"):
        raise ValueError("task 'classification' ya da 'regression' olmali")
    return _BUILDERS[name][task](random_state)


def available_models(task="classification"):
    """Bu bilgisayarda kurulu olan modellerin isimlerini dondurur."""
    names = []
    for name in MODEL_NAMES:
        try:
            get_model(name, task)
            names.append(name)
        except ImportError:
            pass
    return names


def get_models(names=None, task="classification", random_state=42):
    """Isim listesine gore {isim: model} sozlugu dondurur. names=None ise kurulu olanlarin hepsi."""
    if names is None:
        names = available_models(task)
    return {n: get_model(n, task, random_state) for n in names}
