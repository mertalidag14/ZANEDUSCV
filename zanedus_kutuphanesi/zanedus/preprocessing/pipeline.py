"""Tum on isleme adimlarini tek yerde toplayan AutoPreprocessor."""

import numpy as np
import pandas as pd

from ..base import BaseEstimator, TransformerMixin
from ..utils import to_dataframe
from .encoding import OneHotEncoder, OrdinalEncoder
from .imputation import KNNImputer, SimpleImputer
from .outliers import OutlierClipper
from .scaling import SCALERS


class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """Sayisal ve kategorik kolonlari kendisi bulur, uygun adimlari uygular.

    Sayisal: eksik doldur -> aykiri kirp -> olcekle
    Kategorik: eksik doldur -> one-hot / ordinal
    """

    def __init__(self, numeric_impute="median", categorical_impute="most_frequent",
                 outliers="iqr", encoding="onehot", scaling="standard"):
        self.numeric_impute = numeric_impute
        self.categorical_impute = categorical_impute
        self.outliers = outliers
        self.encoding = encoding
        self.scaling = scaling

    def fit(self, X, y=None):
        df = to_dataframe(X)
        self.feature_names_in_ = list(df.columns)
        self.numeric_cols_ = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                              and not pd.api.types.is_bool_dtype(df[c])]
        self.categorical_cols_ = [c for c in df.columns if c not in self.numeric_cols_]

        self.num_steps_ = []
        if self.numeric_cols_:
            num = df[self.numeric_cols_].to_numpy(dtype=float)
            if self.numeric_impute == "knn":
                imputer = KNNImputer()
            else:
                imputer = SimpleImputer(self.numeric_impute)
            steps = [imputer]
            if self.outliers:
                steps.append(OutlierClipper(self.outliers))
            if self.scaling and self.scaling != "none":
                if self.scaling not in SCALERS:
                    raise ValueError(f"Gecersiz scaler: {self.scaling}. Secenekler: {list(SCALERS)}")
                steps.append(SCALERS[self.scaling]())
            for step in steps:
                num = np.asarray(step.fit_transform(num), dtype=float)
            self.num_steps_ = steps

        self.cat_steps_ = []
        if self.categorical_cols_:
            cat = df[self.categorical_cols_].to_numpy(dtype=object)
            fill = "eksik" if self.categorical_impute == "constant" else None
            imputer = SimpleImputer(self.categorical_impute, fill_value=fill)
            if self.encoding == "onehot":
                encoder = OneHotEncoder()
            elif self.encoding == "ordinal":
                encoder = OrdinalEncoder()
            else:
                raise ValueError("encoding 'onehot' ya da 'ordinal' olmali")
            cat = imputer.fit_transform(cat)
            encoder.fit(cat)
            self.cat_steps_ = [imputer, encoder]
        return self

    def transform(self, X):
        df = to_dataframe(X)
        missing = [c for c in self.feature_names_in_ if c not in df.columns]
        if missing:
            raise ValueError(f"Su kolonlar eksik: {missing}")
        parts = []
        if self.numeric_cols_:
            num = df[self.numeric_cols_].to_numpy(dtype=float)
            for step in self.num_steps_:
                num = np.asarray(step.transform(num), dtype=float)
            parts.append(num)
        if self.categorical_cols_:
            cat = df[self.categorical_cols_].to_numpy(dtype=object)
            for step in self.cat_steps_:
                cat = step.transform(cat)
            parts.append(np.asarray(cat, dtype=float))
        return np.hstack(parts)

    def get_feature_names_out(self):
        names = list(self.numeric_cols_)
        if self.categorical_cols_:
            encoder = self.cat_steps_[1]
            if isinstance(encoder, OneHotEncoder):
                names += encoder.get_feature_names_out(self.categorical_cols_)
            else:
                names += list(self.categorical_cols_)
        return names
