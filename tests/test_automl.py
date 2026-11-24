import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from zaneduscv import AutoML, load_model, save_model
from zaneduscv.models import available_models, get_model
from zaneduscv.utils import detect_task

FAST = ["logreg", "knn", "random_forest"]


def test_detect_task():
    assert detect_task([0, 1, 1, 0]) == "classification"
    assert detect_task(["a", "b"]) == "classification"
    assert detect_task(np.linspace(0, 1, 200)) == "regression"


def test_classification_with_string_labels(tmp_path):
    X, y = load_iris(return_X_y=True, as_frame=True)
    y = y.map({0: "setosa", 1: "versicolor", 2: "virginica"})
    automl = AutoML(models=FAST, cv=3, verbose=False).fit(X, y)
    assert automl.task_ == "classification"
    assert automl.best_score_ > 0.9
    preds = automl.predict(X.head())
    assert set(preds) <= {"setosa", "versicolor", "virginica"}
    assert automl.predict_proba(X.head()).shape == (5, 3)
    lb = automl.leaderboard()
    assert list(lb["model"])[0] == automl.best_model_name_

    path = save_model(automl, tmp_path / "model.joblib")
    loaded = load_model(path)
    assert (loaded.predict(X.head()) == preds).all()


def test_regression():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    automl = AutoML(models=["logreg", "random_forest"], cv=3, verbose=False).fit(X, y)
    assert automl.task_ == "regression"
    assert automl.metric_ == "r2"
    assert automl.best_score_ > 0.3


def test_numpy_input_and_categorical():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(size=120),
        "renk": rng.choice(["kirmizi", "mavi"], size=120),
    })
    y = (df["a"] > 0).astype(int)
    automl = AutoML(models=["logreg"], cv=3, verbose=False).fit(df, y)
    assert automl.best_score_ > 0.8
    AutoML(models=["knn"], cv=3, verbose=False).fit(df[["a"]].to_numpy(), y)


def test_unknown_model_name():
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError):
        AutoML(models=["yok_boyle"], verbose=False).fit(X, y)


def test_all_boosters_registered():
    names = available_models()
    for n in ["xgboost", "lightgbm", "catboost"]:
        assert n in names
    assert get_model("catboost", "regression") is not None
