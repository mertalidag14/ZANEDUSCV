# ZANEDUSCV 🤖

ZANEDUSCV, scikit-learn tarzında çalışan kendi AutoML kütüphanem. Veriyi verince ön işlemeyi kendisi yapıyor, model havuzundaki her modeli çapraz doğrulama ile deniyor ve en iyisini seçiyor.

Amaç şu: her yeni veri setinde aynı ön işleme kodunu, aynı model döngüsünü tekrar tekrar yazmamak. `fit` diyorsun, gerisini kütüphane hallediyor.

## 🚀 Hızlı Başlangıç

```python
from zaneduscv import AutoML

automl = AutoML()
automl.fit(X_train, y_train)

automl.leaderboard()        # tüm modellerin sıralaması
automl.predict(X_test)      # en iyi modelle tahmin
```

Breast cancer verisinde çıktı şöyle görünüyor:

```
[ZANEDUSCV] gorev: classification | metrik: accuracy | 8 model deneniyor
  logreg         0.9758 (+/- 0.0082)  0.11s
  knn            0.9670 (+/- 0.0155)  0.17s
  svm            0.9714 (+/- 0.0112)  0.13s
  random_forest  0.9626 (+/- 0.0179)  2.89s
  extra_trees    0.9736 (+/- 0.0149)  2.33s
  xgboost        0.9692 (+/- 0.0162)  0.57s
  lightgbm       0.9670 (+/- 0.0139)  0.52s
  catboost       0.9736 (+/- 0.0164)  2.96s
[ZANEDUSCV] en iyi model: logreg (0.9758)
Test skoru: 0.9825
```

## 📦 Kurulum

```bash
git clone <repo-linki>
cd ZANEDUSCV
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -e ".[all]"
```

XGBoost, LightGBM ve CatBoost kurulu değilse kütüphane yine çalışıyor, sadece o modelleri atlıyor.

## ⚙️ Ne Yapıyor?

**Görev tespiti:** Hedef değişkene bakıp sınıflandırma mı regresyon mu olduğuna kendisi karar veriyor. İstersen `task="regression"` diye elle de verebiliyorsun.

**Ön işleme (`zaneduscv.preprocessing`):**
- Eksik veri: mean, median, most_frequent, KNN
- Aykırı değer: IQR ya da z-skoru ile kırpma
- Encoding: one-hot ya da ordinal
- Scaling: standard, minmax, robust

Sayısal ve kategorik kolonları kendisi buluyor, her birine uygun adımları uyguluyor. Yazı olarak gelen etiketleri (`"evet"`, `"hayir"` gibi) de sorunsuz alıyor.

**Model havuzu (`zaneduscv.models`):**

| Model | Sınıflandırma | Regresyon |
|---|---|---|
| `logreg` | LogisticRegression | Ridge |
| `knn` | KNeighborsClassifier | KNeighborsRegressor |
| `svm` | SVC | SVR |
| `random_forest` | RandomForestClassifier | RandomForestRegressor |
| `extra_trees` | ExtraTreesClassifier | ExtraTreesRegressor |
| `xgboost` | XGBClassifier | XGBRegressor |
| `lightgbm` | LGBMClassifier | LGBMRegressor |
| `catboost` | CatBoostClassifier | CatBoostRegressor |

**Kaydetme:** `save_model` ve `load_model` ile joblib üzerinden kaydedip geri yükleyebiliyorsun.

## 🛠️ Ayarlar

```python
automl = AutoML(
    task="auto",                         # "classification" / "regression"
    models=["xgboost", "lightgbm"],      # sadece bunları dene
    metric="f1_macro",                   # sklearn'deki herhangi bir scoring
    cv=5,
    preprocessing={"scaling": "robust", "outliers": "zscore", "numeric_impute": "knn"},
)
```

## 📁 Klasör Yapısı

```
zaneduscv/
├── core.py                 # AutoML sınıfı
├── preprocessing/          # eksik veri, aykırı değer, encoding, scaling
├── feature_engineering/    # v0.2
├── models/                 # model havuzu
├── optimization/           # v0.3 (Grid, Random, Optuna)
├── ensemble/               # v0.4 (Voting, Bagging, Stacking, Blending)
├── reporting/              # leaderboard, ileride HTML/PDF rapor
└── deployment/             # joblib, ileride FastAPI + Docker
tests/                      # pytest testleri
examples/                   # örnek kullanım
```

## 🗺️ Yol Haritası

- [x] **v0.1** Temel çekirdek, ön işleme, 8 modellik havuz, leaderboard, kaydetme
- [ ] **v0.2** Feature engineering
- [ ] **v0.3** Hiperparametre optimizasyonu (GridSearchCV, RandomizedSearchCV, Optuna)
- [ ] **v0.4** Ensemble (Voting, Bagging, Stacking, Blending)
- [ ] **v0.5** Tam AutoML: SHAP açıklamaları, HTML/PDF rapor
- [ ] **v0.6** FastAPI ve Docker ile dağıtım
- [ ] **v1.0** GPU ve derin öğrenme desteği

## 🧪 Testler

```bash
pytest
```
