import numpy as np
import pandas as pd
import pytest

from zaneduscv import AutoML, load_model, save_model, train_test_split
from zaneduscv.datasets import make_classification, make_regression
from zaneduscv.utils import detect_task

FAST = ["logreg", "knn", "decision_tree", "zboost"]


def test_detect_task():
    assert detect_task([0, 1, 1, 0]) == "classification"
    assert detect_task(["a", "b"]) == "classification"
    assert detect_task(np.linspace(0, 1, 200)) == "regression"


def test_classification_with_string_labels(tmp_path):
    X, y = make_classification(300, n_classes=3, noise=0.0, random_state=0)
    y = y.map({0: "elma", 1: "armut", 2: "kiraz"})
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0, stratify=True)
    automl = AutoML(models=FAST, cv=3, verbose=False).fit(Xtr, ytr)
    assert automl.task_ == "classification"
    assert automl.best_score_ > 0.85
    preds = automl.predict(Xte)
    assert set(preds) <= {"elma", "armut", "kiraz"}
    assert automl.score(Xte, yte) > 0.8
    assert automl.predict_proba(Xte.head()).shape == (5, 3)
    assert list(automl.leaderboard()["model"])[0] == automl.best_model_name_

    path = save_model(automl, tmp_path / "model.pkl")
    assert (load_model(path).predict(Xte) == preds).all()


def test_regression_and_lower_is_better_metric():
    X, y = make_regression(300, random_state=0)
    automl = AutoML(models=["logreg", "zboost"], cv=3, verbose=False).fit(X, y)
    assert automl.task_ == "regression" and automl.metric_ == "r2"
    assert automl.best_score_ > 0.5

    automl = AutoML(models=["logreg", "zboost"], metric="rmse", cv=3, verbose=False).fit(X, y)
    lb = automl.leaderboard()
    assert lb["rmse_mean"].is_monotonic_increasing
    assert automl.best_model_name_ == lb["model"][0]


def test_categorical_and_missing_values():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(size=150),
        "renk": rng.choice(["kirmizi", "mavi", None], size=150),
    })
    y = (df["a"] > 0).astype(int)
    df.loc[::10, "a"] = np.nan
    automl = AutoML(models=["logreg", "zboost"], cv=3, verbose=False).fit(df, y)
    assert automl.best_score_ > 0.75
    AutoML(models=["knn"], cv=3, verbose=False).fit(df[["a"]].fillna(0).to_numpy(), y)


def test_feature_importances():
    X, y = make_classification(200, random_state=0)
    automl = AutoML(models=["zboost"], cv=3, verbose=False).fit(X, y)
    imp = automl.feature_importances()
    assert imp.index[0] in {"f0", "f1", "f2", "f3"}


def test_unknown_model_name():
    X, y = make_classification(50, random_state=0)
    with pytest.raises(ValueError):
        AutoML(models=["yok_boyle"], verbose=False).fit(X, y)
