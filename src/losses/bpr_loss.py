"""
Bayesian Personalized Ranking (BPR) loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BPRLoss(nn.Module):
    """
    BPR loss for pairwise ranking.
    
    Optimizes: log(sigmoid(pos_score - neg_score))
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute BPR loss.
        
        Args:
            pos_logits: [batch_size, seq_len] or [batch_size]
            neg_logits: [batch_size, seq_len] or [batch_size]
            mask: optional mask for valid positions
            
        Returns:
            loss: scalar
        """
        # BPR loss: -log(sigmoid(pos - neg))
        diff = pos_logits - neg_logits
        loss = -F.logsigmoid(diff)
        
        # Apply mask
        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss
        else:
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss


class gBCELoss(nn.Module):
    """
    Generalized BCE loss (gBCE) from gSASRec.
    
    Allows using more negative samples per positive.
    """
    
    def __init__(
        self,
        num_negatives: int = 1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_negatives = num_negatives
        self.reduction = reduction
    
    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gBCE loss.
        
        Args:
            pos_logits: [batch_size, seq_len]
            neg_logits: [batch_size, seq_len, num_neg] or [batch_size, seq_len]
            mask: optional mask
            
        Returns:
            loss: scalar
        """
        # Positive loss
        pos_loss = -F.logsigmoid(pos_logits)
        
        # Negative loss
        if neg_logits.dim() == 3:
            # Multiple negatives
            neg_loss = -F.logsigmoid(-neg_logits).mean(dim=-1)
        else:
            neg_loss = -F.logsigmoid(-neg_logits)
        
        loss = pos_loss + neg_loss
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
            return loss.sum()
        
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
