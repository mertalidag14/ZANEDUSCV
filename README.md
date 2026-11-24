# ZANEDUSCV 🤖

ZANEDUSCV, sıfırdan yazdığım kendi AutoML kütüphanem. scikit-learn, XGBoost, LightGBM ya da CatBoost kullanmıyor; bütün modeller, ön işleme adımları, metrikler ve çapraz doğrulama sadece **numpy** ile yazılı (tablo işleri için pandas var).

Veriyi verince ön işlemeyi kendisi yapıyor, model havuzundaki her modeli çapraz doğrulama ile deniyor ve en iyisini seçiyor. Kullanımı scikit-learn'e benziyor: `fit`, `predict`, `predict_proba`, `score`.

## 🚀 Hızlı Başlangıç

```python
from zaneduscv import AutoML, train_test_split
from zaneduscv.datasets import make_classification

X, y = make_classification(n_samples=800, n_features=10, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=True)

automl = AutoML(cv=5)
automl.fit(X_train, y_train)

automl.leaderboard()        # tüm modellerin sıralaması
automl.predict(X_test)      # en iyi modelle tahmin
```

Çıktı şöyle görünüyor:

```
[ZANEDUSCV] gorev: classification | metrik: accuracy | 7 model deneniyor
  logreg         0.7317 (+/- 0.0347)  0.15s
  knn            0.6441 (+/- 0.0337)  0.09s
  svm            0.6710 (+/- 0.0239)  0.4s
  decision_tree  0.6724 (+/- 0.0144)  0.25s
  random_forest  0.7115 (+/- 0.0270)  9.18s
  extra_trees    0.7192 (+/- 0.0436)  14.88s
  zboost         0.7177 (+/- 0.0298)  1.97s
[ZANEDUSCV] en iyi model: logreg (0.7317)
Test skoru: 0.7233
```

Breast cancer verisinde aynı modelleri denediğimde sonuçlar scikit-learn'ün sonuçlarına çok yakın çıkıyor (SVM %97.7, LogReg %97.4, ZBoost %97.0; sklearn RandomForest %96.3).

## 📦 Kurulum

```bash
git clone https://github.com/mertalidag14/ZANEDUSCV.git
cd ZANEDUSCV
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -e ".[dev]"
```

Tek bağımlılık numpy ve pandas.

## 🧠 Model Havuzu

Hepsini kendim yazdım, her biri hem sınıflandırma hem regresyon yapıyor:

| İsim | Model | Nasıl çalışıyor? |
|---|---|---|
| `logreg` | Lojistik Regresyon / Ridge | Softmax + L2 ceza, Adam ile gradyan inişi. Regresyonda kapalı form çözüm |
| `knn` | K-En Yakın Komşu | Matris işlemleriyle toplu mesafe hesabı |
| `svm` | Destek Vektör Makinesi | RBF çekirdeği Random Fourier Features ile, squared hinge kaybı |
| `decision_tree` | Karar Ağacı (CART) | Gini / MSE, kümülatif toplamlarla hızlı bölme arama |
| `random_forest` | Random Forest | Bootstrap + her bölmede rastgele özellik alt kümesi |
| `extra_trees` | Extra Trees | Eşik değerleri de rastgele, daha hızlı |
| `zboost` | ZBoost | XGBoost mantığı: gradyan + hessian, `G²/(H+λ)` kazancı, erken durdurma |

ZBoost'ta `learning_rate`, `max_depth`, `reg_lambda`, `gamma`, `subsample`, `colsample` ve `early_stopping_rounds` ayarlanabiliyor.

## ⚙️ Ön İşleme

`AutoPreprocessor` sayısal ve kategorik kolonları kendisi buluyor:

- **Eksik veri:** mean, median, most_frequent, constant, KNN
- **Aykırı değer:** IQR ya da z-skoru ile kırpma
- **Encoding:** one-hot ya da ordinal
- **Scaling:** standard, minmax, robust

Yazı olarak gelen etiketleri (`"evet"`, `"hayir"` gibi) de alıyor, tahminleri yine yazı olarak döndürüyor.

## 📏 Metrikler

`accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `roc_auc`, `log_loss`, `r2`, `mse`, `rmse`, `mae`. Küçük olanın iyi olduğu metriklerde (rmse gibi) sıralamayı kendisi ters çeviriyor.

## 🛠️ Ayarlar

```python
automl = AutoML(
    task="auto",                         # "classification" / "regression"
    models=["zboost", "random_forest"],  # sadece bunları dene
    metric="f1_macro",
    cv=5,
    preprocessing={"scaling": "robust", "outliers": "zscore", "numeric_impute": "knn"},
)
automl.fit(X, y)
automl.feature_importances()             # ağaç tabanlı modellerde özellik önemleri
```

Model kaydetme: `save_model(automl, "model.pkl")` ve `load_model("model.pkl")`.

## 📁 Klasör Yapısı

```
zaneduscv/
├── core.py                 # AutoML sınıfı
├── base.py                 # ortak taban sınıflar (get_params, clone ...)
├── metrics.py              # metrikler
├── model_selection.py      # KFold, StratifiedKFold, train_test_split, cross_val_score
├── pipeline.py             # Pipeline
├── datasets.py             # deneme için sentetik veri
├── preprocessing/          # eksik veri, aykırı değer, encoding, scaling
├── models/                 # linear, neighbors, svm, tree, forest, boosting
├── feature_engineering/    # v0.2
├── optimization/           # v0.3
├── ensemble/               # v0.4
├── reporting/              # leaderboard, ileride HTML/PDF rapor
└── deployment/             # pickle ile kaydetme, ileride FastAPI + Docker
tests/                      # pytest testleri
examples/                   # örnek kullanım
```

## 🗺️ Yol Haritası

- [x] **v0.1** Sıfırdan çekirdek: ön işleme, 7 model, metrikler, CV, leaderboard, kaydetme
- [ ] **v0.2** Feature engineering
- [ ] **v0.3** Hiperparametre optimizasyonu (Grid Search, Random Search, Bayesian/TPE)
- [ ] **v0.4** Ensemble (Voting, Bagging, Stacking, Blending)
- [ ] **v0.5** Tam AutoML: SHAP tarzı açıklamalar, HTML/PDF rapor, histogram tabanlı hızlı boosting
- [ ] **v0.6** FastAPI ve Docker ile dağıtım
- [ ] **v1.0** GPU ve derin öğrenme desteği

## 🧪 Testler

```bash
pytest
```
