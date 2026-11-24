"""Model havuzu. Buradaki her model ZANEDUSCV icinde sifirdan yazildi."""

from .boosting import ZBoostClassifier, ZBoostRegressor
from .forest import (ExtraTreesClassifier, ExtraTreesRegressor,
                     RandomForestClassifier, RandomForestRegressor)
from .linear import LogisticRegression, RidgeRegression
from .neighbors import KNNClassifier, KNNRegressor
from .svm import SVMClassifier, SVMRegressor
from .tree import DecisionTreeClassifier, DecisionTreeRegressor

_BUILDERS = {
    "logreg": {
        "classification": lambda s: LogisticRegression(random_state=s),
        "regression": lambda s: RidgeRegression(),
    },
    "knn": {
        "classification": lambda s: KNNClassifier(),
        "regression": lambda s: KNNRegressor(),
    },
    "svm": {
        "classification": lambda s: SVMClassifier(random_state=s),
        "regression": lambda s: SVMRegressor(random_state=s),
    },
    "decision_tree": {
        "classification": lambda s: DecisionTreeClassifier(max_depth=8, min_samples_leaf=2, random_state=s),
        "regression": lambda s: DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, random_state=s),
    },
    "random_forest": {
        "classification": lambda s: RandomForestClassifier(n_estimators=100, random_state=s),
        "regression": lambda s: RandomForestRegressor(n_estimators=100, random_state=s),
    },
    "extra_trees": {
        "classification": lambda s: ExtraTreesClassifier(n_estimators=100, random_state=s),
        "regression": lambda s: ExtraTreesRegressor(n_estimators=100, random_state=s),
    },
    "zboost": {
        "classification": lambda s: ZBoostClassifier(random_state=s),
        "regression": lambda s: ZBoostRegressor(random_state=s),
    },
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
    return list(MODEL_NAMES)


def get_models(names=None, task="classification", random_state=42):
    """Isim listesine gore {isim: model} sozlugu dondurur. names=None ise hepsi."""
    names = MODEL_NAMES if names is None else names
    return {n: get_model(n, task, random_state) for n in names}
