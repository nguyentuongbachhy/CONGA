"""
Binary Cross Entropy loss for sequential recommendation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BCELoss(nn.Module):
    """
    Binary Cross Entropy loss for next-item prediction.
    
    Standard loss used in SASRec.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute BCE loss.
        
        Args:
            pos_logits: [batch_size, seq_len] positive item scores
            neg_logits: [batch_size, seq_len] negative item scores
            mask: [batch_size, seq_len] mask for valid positions
            
        Returns:
            loss: scalar
        """
        # Create labels
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        # Compute BCE
        pos_loss = self.bce(pos_logits, pos_labels)
        neg_loss = self.bce(neg_logits, neg_labels)
        
        loss = pos_loss + neg_loss
        
        # Apply mask
        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / mask.sum()
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
    
    def forward_from_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        pos_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss from model outputs.
        
        Args:
            outputs: Model forward outputs
            pos_items: Positive items (for mask)
            
        Returns:
            loss: scalar
        """
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        mask = (pos_items != 0).float()
        
        return self.forward(pos_logits, neg_logits, mask)
