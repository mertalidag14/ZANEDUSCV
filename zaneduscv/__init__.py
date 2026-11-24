"""ZANEDUSCV: sifirdan yazilmis, egik bolmeli ve lineer yaprakli bir gradient boosting algoritmasi."""

from .model import ZANEDUSCVClassifier, ZANEDUSCVRegressor
from .model_selection import cross_val_score, train_test_split

__version__ = "0.2.0"
__all__ = ["ZANEDUSCVClassifier", "ZANEDUSCVRegressor", "train_test_split", "cross_val_score",
           "__version__"]
