import numpy as np
import pandas as pd
import pytest

from zaneduscv.preprocessing import (AutoPreprocessor, KNNImputer, LabelEncoder, MinMaxScaler,
                                     OneHotEncoder, OutlierClipper, SimpleImputer, StandardScaler)


def test_outlier_clipper_iqr():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    clip = OutlierClipper("iqr").fit(X)
    assert clip.outlier_mask(X)[-1, 0]
    assert clip.transform(X).max() < 100


def test_simple_and_knn_imputer():
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
    out = SimpleImputer("mean").fit_transform(X).astype(float)
    assert out[1, 0] == 2.0 and out[2, 1] == 3.0
    out = KNNImputer(n_neighbors=1).fit_transform(X)
    assert not np.isnan(out).any()


def test_scalers():
    X = np.array([[1.0], [2.0], [3.0]])
    assert np.allclose(StandardScaler().fit_transform(X).mean(), 0)
    mm = MinMaxScaler().fit_transform(X)
    assert mm.min() == 0 and mm.max() == 1


def test_encoders():
    X = np.array([["a"], ["b"], ["a"]])
    enc = OneHotEncoder().fit(X)
    assert enc.transform(np.array([["a"], ["z"]])).tolist() == [[1, 0], [0, 0]]
    le = LabelEncoder().fit(["kedi", "kopek", "kedi"])
    assert list(le.inverse_transform(le.transform(["kopek"]))) == ["kopek"]


def test_autopreprocessor_mixed_types():
    df = pd.DataFrame({
        "yas": [20, 30, np.nan, 40],
        "sehir": ["ankara", None, "izmir", "ankara"],
    })
    prep = AutoPreprocessor().fit(df)
    out = prep.transform(df)
    assert out.shape == (4, 3)
    assert not np.isnan(out).any()
    assert prep.get_feature_names_out() == ["yas", "sehir_ankara", "sehir_izmir"]


def test_bad_scaler():
    with pytest.raises(ValueError):
        AutoPreprocessor(scaling="yok_boyle").fit(pd.DataFrame({"a": [1.0, 2.0]}))
