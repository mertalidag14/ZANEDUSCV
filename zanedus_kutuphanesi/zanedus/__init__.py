"""ZANEDUS: sifirdan yazilmis, sklearn tarzinda bir AutoML kutuphanesi."""

from .core import AutoML
from .deployment import load_model, save_model
from .model_selection import cross_val_score, train_test_split

__version__ = "0.1.0"
__all__ = ["AutoML", "save_model", "load_model", "train_test_split", "cross_val_score",
           "__version__"]
