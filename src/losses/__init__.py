"""
Loss functions for sequential recommendation.
"""

from .bce_loss import BCELoss
from .bpr_loss import BPRLoss
from .duo_loss import DuoLoss
from .infonce import InfoNCELoss
from .distillation import DistillationLoss

__all__ = [
    "BCELoss",
    "BPRLoss", 
    "DuoLoss",
    "InfoNCELoss",
    "DistillationLoss",
]


def get_loss(loss_type: str, **kwargs):
    """Factory function to get loss by name."""
    losses = {
        "bce": BCELoss,
        "bpr": BPRLoss,
        "duo": DuoLoss,
        "infonce": InfoNCELoss,
        "distill": DistillationLoss,
    }
    
    if loss_type.lower() not in losses:
        raise ValueError(f"Unknown loss: {loss_type}. Available: {list(losses.keys())}")
    
    return losses[loss_type.lower()](**kwargs)
