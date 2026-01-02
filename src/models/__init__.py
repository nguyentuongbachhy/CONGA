"""
Model implementations for sequential recommendation.
"""

from .base import BaseModel
from .sasrec import SASRec
from .sasrec_duo import SASRecDuo
from .cl4srec import CL4SRec
from .gcl4sr import GCL4SR
from .conga import CONGA
from .conga_v2 import CONGAv2

__all__ = [
    "BaseModel",
    "SASRec",
    "SASRecDuo", 
    "CL4SRec",
    "GCL4SR",
    "CONGA",
    "CONGAv2",
]


def get_model(model_name: str, **kwargs):
    """Factory function to get model by name."""
    models = {
        "sasrec": SASRec,
        "sasrec_duo": SASRecDuo,
        "cl4srec": CL4SRec,
        "gcl4sr": GCL4SR,
        "conga": CONGA,
        "conga_v2": CONGAv2,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name.lower()](**kwargs)
