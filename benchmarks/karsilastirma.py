"""ZANEDUSCV'yi XGBoost, LightGBM ve CatBoost ile 5 katli capraz dogrulamada karsilastirir.

Calistirmak icin ek paketler lazim (algoritmanin kendisi bunlara ihtiyac duymuyor):
    pip install scikit-learn xgboost lightgbm catboost
    python benchmarks/karsilastirma.py                       # hazir veri setleri
    python benchmarks/karsilastirma.py telefon.csv price_range   # kendi CSV'n
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402

from zaneduscv import ZANEDUSCVClassifier, ZANEDUSCVRegressor  # noqa: E402
from zaneduscv.datasets import make_diagonal  # noqa: E402


def modeller(sinif):
    out = {"ZANEDUSCV": lambda: (ZANEDUSCVClassifier if sinif else ZANEDUSCVRegressor)(random_state=0)}
    try:
        import xgboost as xgb
        out["XGBoost"] = lambda: (xgb.XGBClassifier if sinif else xgb.XGBRegressor)(
            n_estimators=300, learning_rate=0.1, verbosity=0)
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        out["LightGBM"] = lambda: (lgb.LGBMClassifier if sinif else lgb.LGBMRegressor)(verbose=-1)
    except ImportError:
        pass
    try:
        import catboost as cb
        out["CatBoost"] = lambda: (cb.CatBoostClassifier if sinif else cb.CatBoostRegressor)(
            verbose=0, random_seed=0)
    except ImportError:
        pass
    return out


def veri_setleri():
    if len(sys.argv) >= 3:
        df = pd.read_csv(sys.argv[1])
        y = df.pop(sys.argv[2])
        return {sys.argv[1]: (df, y, True)}
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    out = {"meme_kanseri": (X, y, True)}
    X, y = load_wine(return_X_y=True, as_frame=True)
    out["sarap"] = (X, y, True)
    X, y = make_diagonal(random_state=0)
    out["capraz_sinir"] = (X, y, True)
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    out["diyabet (R2)"] = (X, y, False)
    return out


def main():
    satirlar = []
    for ad, (X, y, sinif) in veri_setleri().items():
        y = np.asarray(y)
        cv = (StratifiedKFold if sinif else KFold)(5, shuffle=True, random_state=42)
        for model_adi, yap in modeller(sinif).items():
            skorlar, t0 = [], time.time()
            for tr, te in cv.split(X, y):
                m = yap().fit(X.iloc[tr], y[tr])
                p = np.asarray(m.predict(X.iloc[te])).ravel()
                skorlar.append(accuracy_score(y[te], p) if sinif else r2_score(y[te], p))
            satirlar.append({"veri": ad, "model": model_adi, "skor": np.mean(skorlar),
                             "std": np.std(skorlar), "sure_sn": time.time() - t0})
            print(satirlar[-1], flush=True)
    tablo = pd.DataFrame(satirlar).pivot(index="veri", columns="model", values="skor").round(4)
    print("\n", tablo)


if __name__ == "__main__":
    main()
