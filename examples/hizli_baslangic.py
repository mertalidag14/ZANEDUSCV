"""ZANEDUSCV ile 1 dakikada siniflandirma ve regresyon."""

from zaneduscv import ZANEDUSCVClassifier, ZANEDUSCVRegressor, train_test_split
from zaneduscv.datasets import make_diagonal, make_regression

# --- siniflandirma (capraz sinirli zor veri)
X, y = make_diagonal(n_samples=1500, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=True)

model = ZANEDUSCVClassifier(random_state=0)
model.fit(X_train, y_train)
print("Dogruluk:", round(model.score(X_test, y_test), 4))
print("Kac agac kullanildi:", model.best_iteration_)
print("Kolon onemleri:\n", model.get_feature_importance().round(3))

# --- regresyon
X, y = make_regression(n_samples=800, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
reg = ZANEDUSCVRegressor(random_state=0).fit(X_train, y_train)
print("R2:", round(reg.score(X_test, y_test), 4))

# --- kaydet / yukle
X, y = make_diagonal(n_samples=300, random_state=1)
model.save("zaneduscv_model.pkl")
yuklenen = ZANEDUSCVClassifier.load("zaneduscv_model.pkl")
print("Yuklenen model ayni tahmini yapiyor mu?", bool((yuklenen.predict(X) == model.predict(X)).all()))
