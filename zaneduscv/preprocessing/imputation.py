"""Eksik veri doldurma."""

from sklearn.impute import KNNImputer, SimpleImputer

NUMERIC_STRATEGIES = ("mean", "median", "most_frequent", "knn")
CATEGORICAL_STRATEGIES = ("most_frequent", "constant")


def build_imputer(strategy="median", categorical=False):
    """Istenen stratejiye gore bir imputer dondurur."""
    if categorical:
        if strategy not in CATEGORICAL_STRATEGIES:
            raise ValueError(f"Kategorik icin gecersiz strateji: {strategy}")
        if strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value="eksik")
        return SimpleImputer(strategy="most_frequent")

    if strategy not in NUMERIC_STRATEGIES:
        raise ValueError(f"Sayisal icin gecersiz strateji: {strategy}")
    if strategy == "knn":
        return KNNImputer(n_neighbors=5)
    return SimpleImputer(strategy=strategy)
