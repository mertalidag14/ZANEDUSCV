"""ZANEDUSCV: sklearn tarzinda bir AutoML kutuphanesi."""

from .core import AutoML
from .deployment import load_model, save_model

__version__ = "0.1.0"
__all__ = ["AutoML", "save_model", "load_model", "__version__"]
