"""Veriyi ZANEDUSCV'nin anlayacagi hale getiren katman.

Kullanici DataFrame'i oldugu gibi verebiliyor: yazili kategoriler ve NaN'lar sorun degil.
Burada iki temsil uretiyoruz:
  B : histogram kutulari (uint16). 0 numarali kutu her zaman "eksik / bilinmeyen".
  Z : sayisal kolonlarin standartlastirilmis hali (egik bolmeler ve lineer yapraklar icin).
      Eksik deger 0'a (ortalamaya) cekilir, egitimde gorulen aralik disina tasan degerler kirpilir.
"""

import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, max_bins=64, cat_features=None, max_categories=62):
        self.max_bins = max_bins
        self.cat_features = cat_features
        self.max_categories = max_categories

    # ------------------------------------------------------------------ fit
    def fit(self, X):
        df = self._to_frame(X)
        self.feature_names_ = [str(c) for c in df.columns]
        cat = set()
        if self.cat_features is not None:
            for c in self.cat_features:
                cat.add(self.feature_names_[c] if isinstance(c, (int, np.integer)) else str(c))
        for name, col in zip(self.feature_names_, df.columns):
            s = df[col]
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                cat.add(name)
        self.is_cat_ = np.array([n in cat for n in self.feature_names_])

        self.edges_, self.cat_maps_ = {}, {}
        d = len(self.feature_names_)
        self.mean_, self.std_ = np.zeros(d), np.ones(d)
        self.zmin_, self.zmax_ = np.zeros(d), np.zeros(d)
        self.n_bins_ = np.zeros(d, dtype=int)

        for j, col in enumerate(df.columns):
            s = df[col]
            if self.is_cat_[j]:
                values = s.dropna().astype(str)
                counts = values.value_counts()
                top = list(counts.index[: self.max_categories])
                mapping = {v: i + 1 for i, v in enumerate(top)}
                if len(counts) > len(top):          # seyrek kategoriler ortak "diger" kutusu
                    mapping["__diger__"] = len(top) + 1
                self.cat_maps_[j] = mapping
                self.n_bins_[j] = len(mapping) + 1
            else:
                x = s.to_numpy(dtype=float)
                x = x[~np.isnan(x)]
                if len(x) == 0:
                    self.edges_[j] = np.array([])
                    self.n_bins_[j] = 1
                    continue
                uniq = np.unique(x)
                if len(uniq) <= self.max_bins - 1:
                    edges = (uniq[:-1] + uniq[1:]) / 2
                else:
                    qs = np.quantile(x, np.linspace(0, 1, self.max_bins)[1:-1])
                    edges = np.unique(qs)
                self.edges_[j] = edges
                self.n_bins_[j] = len(edges) + 2
                mu, sd = x.mean(), x.std()
                self.mean_[j], self.std_[j] = mu, (sd if sd > 0 else 1.0)
                z = (x - mu) / self.std_[j]
                self.zmin_[j], self.zmax_[j] = z.min(), z.max()
        self.total_bins_ = int(self.n_bins_.max()) if d else 1
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X):
        df = self._to_frame(X)
        if df.shape[1] != len(self.feature_names_):
            raise ValueError(f"{len(self.feature_names_)} kolon bekleniyordu, {df.shape[1]} geldi")
        n, d = df.shape
        B = np.zeros((n, d), dtype=np.uint16)
        Z = np.zeros((n, d))
        for j, col in enumerate(df.columns):
            s = df[col]
            if self.is_cat_[j]:
                mapping = self.cat_maps_[j]
                other = mapping.get("__diger__", 0)
                vals = s.astype(object).to_numpy()
                for i, v in enumerate(vals):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    B[i, j] = mapping.get(str(v), other)
            else:
                x = s.to_numpy(dtype=float)
                miss = np.isnan(x)
                b = np.searchsorted(self.edges_[j], x, side="left") + 1
                b[miss] = 0
                B[:, j] = b
                z = (x - self.mean_[j]) / self.std_[j]
                z = np.clip(z, self.zmin_[j], self.zmax_[j])
                z[miss] = 0.0
                Z[:, j] = z
        return B, Z

    @staticmethod
    def _to_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
