"""Model kaydetme / yukleme. FastAPI ve Docker destegi sonraki surumlerde geliyor."""

from .persistence import load_model, save_model

__all__ = ["save_model", "load_model"]
