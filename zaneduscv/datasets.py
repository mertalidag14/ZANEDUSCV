"""Deneme icin sentetik veri uretici (disaridan veri setine ihtiyac olmadan)."""

import numpy as np
import pandas as pd


def make_classification(n_samples=500, n_features=8, n_informative=4, n_classes=2,
                        noise=0.1, random_state=None, as_frame=True):
    """Her sinif icin farkli bir merkezden Gauss bulutlari uretir."""
    rng = np.random.default_rng(random_state)
    centers = rng.normal(scale=2.0, size=(n_classes, n_informative))
    y = rng.integers(0, n_classes, n_samples)
    informative = centers[y] + rng.normal(size=(n_samples, n_informative))
    extra = rng.normal(size=(n_samples, n_features - n_informative))
    X = np.hstack([informative, extra])
    flip = rng.random(n_samples) < noise
    y[flip] = rng.integers(0, n_classes, flip.sum())
    if as_frame:
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), pd.Series(y, name="hedef")
    return X, y


def make_regression(n_samples=500, n_features=8, noise=0.5, nonlinear=True,
                    random_state=None, as_frame=True):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    y = X @ w
    if nonlinear:
        y += 2 * np.sin(X[:, 0] * 2) + X[:, 1] ** 2
    y += rng.normal(scale=noise, size=n_samples)
    if as_frame:
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), pd.Series(y, name="hedef")
    return X, y


def make_diagonal(n_samples=1500, noise=0.05, random_state=None, as_frame=True):
    """Sinirlari capraz (eksenlere egik) olan zor bir siniflandirma verisi.

    Normal karar agaclari bu siniri merdiven gibi parca parca ogrenmek zorunda kaliyor,
    ZANEDUSCV'nin egik bolmeleri ise tek adimda yakalayabiliyor.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 6))
    y = ((X[:, 0] + X[:, 1] > 0) ^ (X[:, 2] - 0.7 * X[:, 3] > 0.3)).astype(int)
    flip = rng.random(n_samples) < noise
    y[flip] = 1 - y[flip]
    if as_frame:
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(6)]), pd.Series(y, name="hedef")
    return X, y
