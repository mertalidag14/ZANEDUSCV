import numpy as np

from zaneduscv.metrics import accuracy, f1_macro, mae, r2, rmse


def test_metrikler():
    assert accuracy([0, 1, 1], [0, 1, 0]) == 2 / 3
    assert r2([1, 2, 3], [1, 2, 3]) == 1.0
    assert rmse([0, 0], [3, 4]) == np.sqrt(12.5)
    assert mae([1, 2], [2, 4]) == 1.5
    assert 0 <= f1_macro([0, 1, 1, 0], [0, 1, 0, 0]) <= 1
