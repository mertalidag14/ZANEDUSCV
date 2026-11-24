"""ZANEDUSCV hizli baslangic ornegi. Disaridan hicbir ML kutuphanesi gerekmiyor."""

from zaneduscv import AutoML, save_model, train_test_split
from zaneduscv.datasets import make_classification

X, y = make_classification(n_samples=800, n_features=10, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)

automl = AutoML(cv=5)
automl.fit(X_train, y_train)

print(automl.leaderboard())
print("Test skoru:", round(automl.score(X_test, y_test), 4))

save_model(automl, "models/zaneduscv_model.pkl")
