"""Olceklendirme."""

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def build_scaler(strategy="standard"):
    if strategy is None or strategy == "none":
        return "passthrough"
    if strategy not in SCALERS:
        raise ValueError(f"Gecersiz scaler: {strategy}. Secenekler: {list(SCALERS)}")
    return SCALERS[strategy]()
