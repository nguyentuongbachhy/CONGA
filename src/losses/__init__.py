"""
Loss functions for sequential recommendation.
"""

from .bce_loss import BCELoss
from .bpr_loss import BPRLoss
from .duo_loss import DuoLoss
from .infonce import InfoNCELoss
from .distillation import DistillationLoss
from .listmle_loss import ListMLELoss, ListMLELossSimplified
from .neuralndcg_loss import NeuralNDCGLoss, ApproxNDCGLoss

__all__ = [
    "BCELoss",
    "BPRLoss", 
    "DuoLoss",
    "InfoNCELoss",
    "DistillationLoss",
    "ListMLELoss",
    "ListMLELossSimplified",
    "NeuralNDCGLoss",
    "ApproxNDCGLoss",
]


def get_loss(loss_type: str, **kwargs):
    """Factory function to get loss by name."""
    losses = {
        "bce": BCELoss,
        "bpr": BPRLoss,
        "duo": DuoLoss,
        "infonce": InfoNCELoss,
        "distill": DistillationLoss,
        "listmle": ListMLELoss,
        "listmle_simple": ListMLELossSimplified,
        "neuralndcg": NeuralNDCGLoss,
        "approxndcg": ApproxNDCGLoss,
    }
    
    if loss_type.lower() not in losses:
        raise ValueError(f"Unknown loss: {loss_type}. Available: {list(losses.keys())}")
    
    return losses[loss_type.lower()](**kwargs)
