import numpy as np
import pytest

from zaneduscv.datasets import make_classification, make_regression
from zaneduscv.metrics import accuracy, f1_macro, r2, roc_auc
from zaneduscv.model_selection import StratifiedKFold, train_test_split
from zaneduscv.models import MODEL_NAMES, get_model
from zaneduscv.preprocessing import StandardScaler


def _split(task):
    if task == "classification":
        X, y = make_classification(400, n_classes=3, noise=0.0, random_state=1, as_frame=False)
    else:
        X, y = make_regression(400, random_state=1, as_frame=False)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_classifiers_learn(name):
    Xtr, Xte, ytr, yte = _split("classification")
    model = get_model(name, "classification").fit(Xtr, ytr)
    assert accuracy(yte, model.predict(Xte)) > 0.75
    proba = model.predict_proba(Xte)
    assert proba.shape == (len(Xte), 3)
    assert np.allclose(proba.sum(axis=1), 1)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_regressors_learn(name):
    Xtr, Xte, ytr, yte = _split("regression")
    model = get_model(name, "regression").fit(Xtr, ytr)
    assert r2(yte, model.predict(Xte)) > 0.4


def test_tree_importances_find_signal():
    X, y = make_classification(300, n_features=6, n_informative=2, noise=0.0,
                               random_state=3, as_frame=False)
    model = get_model("random_forest").fit(X, y)
    top2 = set(np.argsort(model.feature_importances_)[-2:])
    assert top2 == {0, 1}


def test_metrics():
    assert accuracy([1, 0, 1], [1, 1, 1]) == pytest.approx(2 / 3)
    assert f1_macro([0, 1], [0, 1]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert r2([1, 2, 3], [1, 2, 3]) == 1.0


def test_stratified_kfold_keeps_ratio():
    y = np.array([0] * 80 + [1] * 20)
    for tr, val in StratifiedKFold(5, random_state=0).split(y, y):
        assert len(set(tr) & set(val)) == 0
        assert y[val].mean() == pytest.approx(0.2)


def test_unknown_model():
    with pytest.raises(ValueError):
        get_model("yok_boyle")
