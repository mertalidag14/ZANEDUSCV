"""ZANEDUSCV hizli baslangic ornegi."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from zaneduscv import AutoML, save_model

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

automl = AutoML(cv=5)
automl.fit(X_train, y_train)

print(automl.leaderboard())
print("Test skoru:", round(automl.score(X_test, y_test), 4))

save_model(automl, "models/zaneduscv_model.joblib")
