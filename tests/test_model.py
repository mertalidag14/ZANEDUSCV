import numpy as np
import pandas as pd
import pytest

from zaneduscv import ZANEDUSCVClassifier, ZANEDUSCVRegressor, cross_val_score, train_test_split
from zaneduscv.base import clone
from zaneduscv.datasets import make_classification, make_diagonal, make_regression

HIZLI = dict(n_estimators=60, learning_rate=0.1, random_state=0)


def test_ikili_siniflandirma():
    X, y = make_classification(n_samples=400, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=True)
    m = ZANEDUSCVClassifier(**HIZLI).fit(Xtr, ytr)
    assert m.score(Xte, yte) > 0.8
    p = m.predict_proba(Xte)
    assert p.shape == (len(Xte), 2)
    assert np.allclose(p.sum(axis=1), 1)


def test_cok_sinifli_ve_yazili_etiket():
    X, y = make_classification(n_samples=450, n_classes=3, random_state=1)
    etiket = np.array(["dusuk", "orta", "yuksek"])[y]
    m = ZANEDUSCVClassifier(**HIZLI).fit(X, etiket)
    assert set(m.classes_) == {"dusuk", "orta", "yuksek"}
    assert m.predict_proba(X).shape == (450, 3)
    assert m.score(X, etiket) > 0.8


def test_regresyon():
    X, y = make_regression(n_samples=400, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    m = ZANEDUSCVRegressor(**HIZLI).fit(Xtr, ytr)
    assert m.score(Xte, yte) > 0.7


def test_kategorik_ve_eksik_deger():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=400), "renk": rng.choice(["kirmizi", "mavi", "yesil"], 400)})
    df.loc[::6, "a"] = np.nan
    y = ((df.renk == "mavi") | (df.a > 0.5)).astype(int)
    m = ZANEDUSCVClassifier(**HIZLI).fit(df, y)
    assert m.score(df, y) > 0.9
    yeni = pd.DataFrame({"a": [np.nan, 1.0], "renk": ["mor", None]})   # gorulmemis kategori + eksik
    assert m.predict(yeni).shape == (2,)


def test_egik_bolme_fark_yaratiyor():
    X, y = make_diagonal(n_samples=1200, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=True)
    egik = ZANEDUSCVClassifier(**HIZLI).fit(Xtr, ytr).score(Xte, yte)
    duz = ZANEDUSCVClassifier(oblique=False, **HIZLI).fit(Xtr, ytr).score(Xte, yte)
    assert egik > duz


def test_erken_durdurma_ve_eval_set():
    X, y = make_classification(n_samples=500, random_state=2)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=0)
    m = ZANEDUSCVClassifier(n_estimators=400, learning_rate=0.3, early_stopping_rounds=10,
                            random_state=0).fit(Xtr, ytr, eval_set=(Xv, yv))
    assert m.best_iteration_ < 400


def test_numpy_girdi_ve_onem():
    X, y = make_classification(n_samples=300, random_state=3, as_frame=False)
    m = ZANEDUSCVClassifier(**HIZLI).fit(X, y)
    assert m.feature_importances_.shape == (X.shape[1],)
    assert abs(m.feature_importances_.sum() - 1) < 1e-6
    assert len(m.get_feature_importance()) == X.shape[1]


def test_kaydet_yukle(tmp_path):
    X, y = make_regression(n_samples=200, random_state=4)
    m = ZANEDUSCVRegressor(**HIZLI).fit(X, y)
    yol = tmp_path / "model.pkl"
    m.save(yol)
    m2 = ZANEDUSCVRegressor.load(yol)
    assert np.allclose(m.predict(X), m2.predict(X))


def test_ayni_seed_ayni_sonuc():
    X, y = make_classification(n_samples=300, random_state=5)
    p1 = ZANEDUSCVClassifier(**HIZLI).fit(X, y).predict_proba(X)
    p2 = ZANEDUSCVClassifier(**HIZLI).fit(X, y).predict_proba(X)
    assert np.allclose(p1, p2)


def test_clone_ve_cross_val():
    X, y = make_classification(n_samples=300, random_state=6)
    m = ZANEDUSCVClassifier(**HIZLI)
    assert clone(m).get_params() == m.get_params()
    skorlar = cross_val_score(m, X, y, cv=3, random_state=0)
    assert len(skorlar) == 3 and skorlar.mean() > 0.75


def test_hatali_girdi():
    with pytest.raises(ValueError):
        ZANEDUSCVClassifier(**HIZLI).fit(np.zeros((10, 2)), np.zeros(10))
    X, y = make_classification(n_samples=200, random_state=7)
    m = ZANEDUSCVClassifier(**HIZLI).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X.drop(columns="f0"))
