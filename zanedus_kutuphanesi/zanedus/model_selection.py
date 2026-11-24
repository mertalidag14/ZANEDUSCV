"""Veri bolme ve capraz dogrulama."""

import numpy as np

from .base import clone
from .metrics import get_metric


class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits en az 2 olmali")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(idx)
        for fold in np.array_split(idx, self.n_splits):
            yield np.setdiff1d(idx, fold), fold


class StratifiedKFold(KFold):
    """Her katta sinif oranlarini korur."""

    def split(self, X, y):
        y = np.asarray(y).ravel()
        rng = np.random.default_rng(self.random_state)
        folds = [[] for _ in range(self.n_splits)]
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if self.shuffle:
                rng.shuffle(cls_idx)
            for i, part in enumerate(np.array_split(cls_idx, self.n_splits)):
                folds[i].extend(part)
        all_idx = np.arange(len(y))
        for fold in folds:
            fold = np.sort(np.array(fold, dtype=int))
            yield np.setdiff1d(all_idx, fold), fold


def train_test_split(X, y, test_size=0.2, random_state=None, stratify=False):
    """X ve y'yi egitim/test olarak boler. X DataFrame de olabilir."""
    y_arr = np.asarray(y).ravel()
    n = len(y_arr)
    rng = np.random.default_rng(random_state)
    if stratify:
        test_idx = []
        for cls in np.unique(y_arr):
            cls_idx = np.where(y_arr == cls)[0]
            rng.shuffle(cls_idx)
            test_idx.extend(cls_idx[: max(1, int(round(len(cls_idx) * test_size)))])
        test_idx = np.array(sorted(test_idx))
    else:
        perm = rng.permutation(n)
        test_idx = np.sort(perm[: int(round(n * test_size))])
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    return _take(X, train_idx), _take(X, test_idx), _take(y, train_idx), _take(y, test_idx)


def _take(data, idx):
    if hasattr(data, "iloc"):
        return data.iloc[idx]
    return np.asarray(data)[idx]


def score_estimator(estimator, X, y, metric):
    func, _, kind = get_metric(metric)
    if kind == "proba":
        return func(y, estimator.predict_proba(X))
    return func(y, estimator.predict(X))


def cross_val_score(estimator, X, y, cv=5, metric="accuracy", random_state=None):
    """Her katta modeli sifirdan egitip skorlari dondurur."""
    y = np.asarray(y).ravel()
    if isinstance(cv, int):
        splitter_cls = StratifiedKFold if getattr(estimator, "task", None) == "classification" else KFold
        cv = splitter_cls(cv, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        model = clone(estimator)
        model.fit(_take(X, train_idx), y[train_idx])
        scores.append(score_estimator(model, _take(X, val_idx), y[val_idx], metric))
    return np.array(scores)
