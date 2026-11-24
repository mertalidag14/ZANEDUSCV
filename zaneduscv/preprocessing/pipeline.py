"""Tum on isleme adimlarini tek yerde toplayan AutoPreprocessor."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..utils import to_dataframe
from .encoding import build_encoder
from .imputation import build_imputer
from .outliers import OutlierClipper
from .scaling import build_scaler


class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """Sayisal ve kategorik kolonlari otomatik bulup uygun adimlari uygular."""

    def __init__(self, numeric_impute="median", categorical_impute="most_frequent",
                 outliers="iqr", encoding="onehot", scaling="standard"):
        self.numeric_impute = numeric_impute
        self.categorical_impute = categorical_impute
        self.outliers = outliers
        self.encoding = encoding
        self.scaling = scaling

    def _build(self, X):
        self.numeric_cols_ = X.select_dtypes(include="number").columns.tolist()
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        num_steps = [("imputer", build_imputer(self.numeric_impute))]
        if self.outliers:
            num_steps.append(("outliers", OutlierClipper(method=self.outliers)))
        num_steps.append(("scaler", build_scaler(self.scaling)))

        cat_steps = [
            ("imputer", build_imputer(self.categorical_impute, categorical=True)),
            ("encoder", build_encoder(self.encoding)),
        ]

        transformers = []
        if self.numeric_cols_:
            transformers.append(("num", Pipeline(num_steps), self.numeric_cols_))
        if self.categorical_cols_:
            transformers.append(("cat", Pipeline(cat_steps), self.categorical_cols_))
        return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    def fit(self, X, y=None):
        X = to_dataframe(X)
        self.feature_names_in_ = X.columns.tolist()
        self.transformer_ = self._build(X).fit(X, y)
        return self

    def transform(self, X):
        X = to_dataframe(X)
        return self.transformer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.transformer_.get_feature_names_out()
