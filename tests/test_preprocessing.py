import numpy as np
import pandas as pd
import pytest

from zaneduscv.preprocessing import AutoPreprocessor, OutlierClipper, build_scaler


def test_outlier_clipper_iqr():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    clip = OutlierClipper("iqr").fit(X)
    assert clip.outlier_mask(X)[-1, 0]
    assert clip.transform(X).max() < 100


def test_autopreprocessor_mixed_types():
    df = pd.DataFrame({
        "yas": [20, 30, np.nan, 40],
        "sehir": ["ankara", None, "izmir", "ankara"],
    })
    out = AutoPreprocessor().fit_transform(df)
    assert out.shape[0] == 4
    assert not np.isnan(out).any()


def test_bad_scaler():
    with pytest.raises(ValueError):
        build_scaler("yok_boyle")
