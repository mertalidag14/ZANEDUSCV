"""Basari metrikleri, hepsi numpy ile."""

import numpy as np


def _arr(y):
    return np.asarray(y).ravel()


# ---------------- siniflandirma ----------------

def accuracy(y_true, y_pred):
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm


def _prf(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0)
    real_pos = cm.sum(axis=1)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, real_pos, out=np.zeros_like(tp), where=real_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def precision_macro(y_true, y_pred):
    return float(_prf(y_true, y_pred)[0].mean())


def recall_macro(y_true, y_pred):
    return float(_prf(y_true, y_pred)[1].mean())


def f1_macro(y_true, y_pred):
    return float(_prf(y_true, y_pred)[2].mean())


def log_loss(y_true, proba, eps=1e-15):
    """y_true 0..k-1 tam sayi etiketleri olmali."""
    y_true = _arr(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), eps, 1 - eps)
    return float(-np.mean(np.log(proba[np.arange(len(y_true)), y_true])))


def roc_auc(y_true, scores):
    """Ikili siniflandirma icin AUC (Mann-Whitney U ile, siralama tabanli)."""
    y_true = _arr(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 2:
        scores = scores[:, 1]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc icin iki sinif da bulunmali")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # esit skorlara ortalama sira ver
    for s in np.unique(scores):
        mask = scores == s
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------- regresyon ----------------

def mse(y_true, y_pred):
    y_true, y_pred = _arr(y_true).astype(float), _arr(y_pred).astype(float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    y_true, y_pred = _arr(y_true).astype(float), _arr(y_pred).astype(float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    y_true, y_pred = _arr(y_true).astype(float), _arr(y_pred).astype(float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------- AutoML icin metrik tablosu ----------------
# (fonksiyon, buyuk mu iyi, tahmin turu: "label" / "proba")
METRICS = {
    "accuracy": (accuracy, True, "label"),
    "f1_macro": (f1_macro, True, "label"),
    "precision_macro": (precision_macro, True, "label"),
    "recall_macro": (recall_macro, True, "label"),
    "roc_auc": (roc_auc, True, "proba"),
    "log_loss": (log_loss, False, "proba"),
    "r2": (r2, True, "label"),
    "mse": (mse, False, "label"),
    "rmse": (rmse, False, "label"),
    "mae": (mae, False, "label"),
}


def get_metric(name):
    if name not in METRICS:
        raise ValueError(f"Bilinmeyen metrik: {name}. Secenekler: {list(METRICS)}")
    return METRICS[name]
