# ZANEDUSCV 🌳⚡

> ⚠️ **ZANEDUSCV bir kütüphane değil, bir makine öğrenmesi algoritması.**
> İçinde hazır modeller toplayan bir araç değil; CatBoost, XGBoost ya da LightGBM gibi **tek başına çalışan kendi algoritmam**. Sıfırdan, sadece **numpy** ile yazılı (tablo işleri için pandas kullanıyor).

👤 **Geliştiren:** Mert Ali Dağ ([@mertalidag14](https://github.com/mertalidag14)). ZANEDUSCV bir ekip işi değil, tek başıma geliştirdiğim kişisel projem.

## 📁 Repoda Neler Var?

| Klasör | Ne? |
|---|---|
| ⭐ [`zaneduscv/`](zaneduscv) | **Asıl proje: ZANEDUSCV algoritması.** Bu README onu anlatıyor. |
| 🧰 [`zanedus_kutuphanesi/`](zanedus_kutuphanesi) | Yan parça: daha önce yazdığım küçük AutoML kütüphanesi (lojistik regresyon, KNN, SVM, karar ağacı, random forest ...). Kendi README'si içinde. |

## 📱 Nerede Kullanılıyor? TeknoTahmin

ZANEDUSCV'yi gerçek bir projede de kullanıyorum: **TeknoTahmin**, telefonların özelliklerine bakıp fiyat aralığını tahmin eden okul projesi. TeknoTahmin bir ekip projesi; içinde kullanılan ZANEDUSCV algoritmasını ise sadece ben geliştirdim.

- 🔗 Repo: [mertalidag14/Mobile-Price-Range](https://github.com/mertalidag14/Mobile-Price-Range)
- 🌐 Canlı site: [teknotahmin-0iji.onrender.com](https://teknotahmin-0iji.onrender.com)

TeknoTahmin'deki gerçek telefon verisinde (Smartprix'ten 970 telefon, sitedeki 10 donanım özelliği, 4 fiyat segmenti) 5 katlı çapraz doğrulama sonuçları:

| Model | Doğruluk |
|---|---|
| CatBoost | **%77.1** |
| **ZANEDUSCV (`auto_tune=True`)** | %77.0 |
| Random Forest | %76.7 |
| LightGBM | %75.8 |
| ZANEDUSCV (varsayılan ayar) | %75.7 |
| XGBoost | %75.7 |

Otomatik ayar modu açıkken ZANEDUSCV, CatBoost'la neredeyse berabere geliyor (970 telefonda yaklaşık 1 telefonluk fark). Ayar araması her katın içinde ayrı yapılıyor, yani test verisine bakarak ayar seçilmiyor.

## 🌳 ZANEDUSCV Nasıl Çalışıyor?

ZANEDUSCV bir **gradient boosting** algoritması: ağaçları art arda ekliyor ve her yeni ağaç öncekilerin hatasını düzeltmeye çalışıyor. Buraya kadarı bildiğimiz boosting. Farkı, ağaçların nasıl kurulduğunda.

## 💡 Diğerlerinden Farkı Ne?

Klasik ağaçlar (XGBoost, LightGBM, CatBoost) her adımda **tek bir kolona** bakıp "x < 5 mi?" diye soruyor ve yapraklarına **sabit bir sayı** koyuyor. ZANEDUSCV'de dört şey farklı:

| # | Fikir | Ne işe yarıyor |
|---|---|---|
| 1 | **Eğik bölmeler** | Ağaç gerekirse iki kolonu birlikte kullanıyor: `w1·x1 + w2·x2 ≤ eşik`. Ağırlıklar o düğümdeki gradyandan, ağırlıklı en küçük karelerle hesaplanıyor. Böylece çapraz bir sınırı merdiven gibi onlarca bölmeyle değil, tek bölmeyle yakalıyor. |
| 2 | **Lineer yapraklar** | Her yaprakta sabit bir değer yerine küçük bir ridge regresyonu var: `b + w·x`. Kullandığı kolonlar o yaprağa giden yoldaki kolonlar. Yaprağın içindeki eğimi de öğreniyor. |
| 3 | **Güven katsayısı** | Yaprağın çıktısı `n / (n + κ)` ile çarpılıyor. Az örnekli yapraklar modeli daha az etkiliyor, böylece ezberleme azalıyor. |
| 4 | **Doğal kategorik ve eksik değer desteği** | Yazılı kategorileri ve `NaN` değerleri olduğu gibi veriyorsun. Eksik değerler için hangi yöne gideceğini ağaç kendisi öğreniyor; kategoriler gradyan oranına göre sıralanıp en iyi gruplama bulunuyor. |

Bunların üstünde ikinci dereceden (Newton) adımlar, histogram tabanlı bölme arama, histogram çıkarma hilesi, satır/kolon örnekleme ve erken durdurma var.

## 📊 Sonuçlar

5 katlı çapraz doğrulama ile, her modelin kendi varsayılan ayarlarıyla (doğruluk; diyabette R²):

| Veri seti | **ZANEDUSCV** | CatBoost | XGBoost | LightGBM |
|---|---|---|---|---|
| Meme kanseri | **%97.4** | %96.5 | %96.5 | %96.7 |
| Şarap | **%97.7** | **%97.7** | %96.1 | %96.6 |
| Telefon fiyat aralığı (Kaggle) | %93.6 | **%93.7** | %91.1 | %90.5 |
| Çapraz sınır (sentetik) | **%90.3** | %82.2 | %80.1 | %79.7 |
| Diyabet (R²) | **0.464** | 0.430 | 0.328 | 0.401 |

Dürüst olmak gerekirse:
- 🏆 5 veri setinin 4'ünde en iyi ya da berabere, telefon verisinde CatBoost çok az önde.
- 🐢 Saf Python/numpy olduğu için diğerlerinden **daha yavaş** (onlar C++ ile yazılı).
- Küçük veri setlerinde farklar birkaç örneğe denk geliyor, o yüzden "her zaman en iyi" demiyorum.

Hangi fikrin ne kattığını da tek tek kapatarak ölçtüm (ablation):

| Veri seti | Tam model | Eğik bölme yok | Lineer yaprak yok | Hepsi kapalı (klasik boosting) |
|---|---|---|---|---|
| Çapraz sınır | %90.3 | %80.5 | %90.6 | %79.9 |
| Telefon fiyatı | %93.6 | %91.6 | %93.7 | %91.0 |
| Diyabet (R²) | 0.464 | 0.441 | 0.463 | 0.428 |

En büyük farkı eğik bölmeler yaratıyor 🎯

Karşılaştırmayı kendin çalıştırmak için:

```bash
pip install scikit-learn xgboost lightgbm catboost
python benchmarks/karsilastirma.py
python benchmarks/karsilastirma.py telefon.csv price_range    # kendi CSV dosyanla
```

## 🚀 Hızlı Başlangıç

```python
from zaneduscv import ZANEDUSCVClassifier, train_test_split
from zaneduscv.datasets import make_diagonal

X, y = make_diagonal(n_samples=1500, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=True)

model = ZANEDUSCVClassifier(random_state=0)
model.fit(X_train, y_train)

model.predict(X_test)            # sınıf tahmini
model.predict_proba(X_test)      # olasılıklar
model.score(X_test, y_test)      # doğruluk
model.get_feature_importance()   # kolon önemleri
```

Regresyon için `ZANEDUSCVRegressor` var, kullanımı aynı.

Kategorik kolonlar ve eksik değerler için ekstra bir şey yapmana gerek yok:

```python
import pandas as pd

df = pd.DataFrame({"ram": [4, 8, None, 6], "marka": ["A", "B", "A", None]})
model.fit(df, ["ucuz", "pahali", "ucuz", "orta"])
```

Ayarlarla uğraşmak istemiyorsan otomatik ayar modunu açabilirsin. Eğitim birkaç kat uzun sürüyor ama en iyi ayarı kendisi buluyor:

```python
model = ZANEDUSCVClassifier(auto_tune=True, random_state=0).fit(X_train, y_train)
model.best_params_     # seçilen ayarlar
model.tune_results_    # denenen bütün ayarlar ve skorları
```

Kaydetme ve yükleme:

```python
model.save("model.pkl")
model = ZANEDUSCVClassifier.load("model.pkl")
```

## ⚙️ Önemli Parametreler

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `n_estimators` | 500 | En fazla kaç tur ağaç eklenecek |
| `learning_rate` | 0.05 | Her ağacın katkısı |
| `max_depth` | 5 | Ağaç derinliği |
| `oblique` | True | Eğik bölmeleri aç/kapat |
| `oblique_top` | 3 | Eğik bölme için denenecek en iyi kolon sayısı |
| `linear_leaves` | True | Lineer yaprakları aç/kapat |
| `linear_lambda` | 1.0 | Lineer yaprakların ridge cezası |
| `confidence_kappa` | 10.0 | Güven katsayısındaki κ (0 = kapalı) |
| `reg_lambda` | 1.0 | L2 düzenlileştirme |
| `subsample`, `colsample` | 0.8, 0.8 | Satır ve kolon örnekleme oranı |
| `early_stopping_rounds` | 50 | Doğrulama kaybı bu kadar tur düzelmezse duruyor |
| `validation_fraction` | 0.1 | `eval_set` verilmezse erken durdurma için ayrılan oran |
| `max_bins` | 64 | Sayısal kolonların histogram kutusu sayısı |
| `auto_tune` | False | True ise farklı ayarları kendi içinde çapraz doğrulamayla deneyip en iyisini seçiyor |

`fit(X, y, cat_features=None, eval_set=None)`: `cat_features` ile sayı gibi görünen ama aslında kategorik olan kolonları belirtebiliyorsun.

## 📦 Kurulum

```bash
git clone https://github.com/mertalidag14/ZANEDUSCV.git
cd ZANEDUSCV
pip install -e .
```

Tek bağımlılık numpy ve pandas.

## 🗂️ Proje Yapısı

```
zaneduscv/              # ⭐ ASIL PROJE: ZANEDUSCV algoritması
├── _data.py            # veriyi histogram kutularına ve standart değerlere çeviriyor
├── _tree.py            # ZANEDUSCV ağacı: eğik bölme, lineer yaprak, güven katsayısı
├── _booster.py         # boosting döngüsü ve kayıp fonksiyonları
├── model.py            # ZANEDUSCVClassifier / ZANEDUSCVRegressor
├── base.py             # fit / predict / get_params altyapısı
├── metrics.py          # doğruluk, F1, log loss, R², RMSE ...
├── model_selection.py  # train_test_split, KFold, cross_val_score
└── datasets.py         # deneme verisi üreticileri
benchmarks/             # XGBoost, LightGBM, CatBoost ile karşılaştırma
examples/               # örnek kullanım
tests/                  # pytest testleri
zanedus_kutuphanesi/    # 🧰 yan parça: ZANEDUS AutoML kütüphanesi (eski modeller)
```

## 🗺️ Yol Haritası

- [x] Eğik bölmeler, lineer yapraklar, güven katsayısı
- [x] Kategorik ve eksik değer desteği, erken durdurma
- [x] Kendi kendini ayarlayan mod (`auto_tune=True`)
- [x] scikit-learn uyumu (`GridSearchCV`, `cross_val_score`)
- [ ] Hız: kritik döngüleri derlenmiş koda taşımak

## 🧪 Testler

```bash
pip install pytest
pytest
```
