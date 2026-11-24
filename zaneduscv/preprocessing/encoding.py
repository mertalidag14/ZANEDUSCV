"""Kategorik degiskenleri sayiya cevirme."""

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_encoder(strategy="onehot"):
    if strategy == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)
    if strategy == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError("strategy 'onehot' ya da 'ordinal' olmali")
