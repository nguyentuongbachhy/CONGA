"""
DuoRec loss: Contrastive regularization for sequential recommendation.
Based on: "Contrastive Learning for Representation Degeneration Problem" (WWW 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DuoLoss(nn.Module):
    """
    DuoRec loss combining:
    1. Unsupervised contrastive loss (between two views)
    2. Supervised contrastive loss (based on target items)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        unsup_weight: float = 0.1,
        sup_weight: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.unsup_weight = unsup_weight
        self.sup_weight = sup_weight
    
    def unsupervised_contrastive_loss(
        self,
        repr_1: torch.Tensor,
        repr_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute unsupervised contrastive loss (InfoNCE).
        
        Args:
            repr_1: [batch_size, hidden_size] first view
            repr_2: [batch_size, hidden_size] second view
            
        Returns:
            loss: scalar
        """
        batch_size = repr_1.shape[0]
        
        # Normalize
        z1 = F.normalize(repr_1, dim=-1)
        z2 = F.normalize(repr_2, dim=-1)
        
        # Similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Labels: diagonal pairs
        labels = torch.arange(batch_size, device=repr_1.device)
        
        # Symmetric loss
        loss = (
            F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)
        ) / 2
        
        return loss
    
    def supervised_contrastive_loss(
        self,
        repr_1: torch.Tensor,
        repr_2: torch.Tensor,
        target_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Sequences with the same target item are treated as positive pairs.
        
        Args:
            repr_1: [batch_size, hidden_size]
            repr_2: [batch_size, hidden_size]
            target_items: [batch_size] target item indices
            
        Returns:
            loss: scalar
        """
        batch_size = repr_1.shape[0]
        device = repr_1.device
        
        # Normalize
        z1 = F.normalize(repr_1, dim=-1)
        z2 = F.normalize(repr_2, dim=-1)
        
        # Combined representations
        z = torch.cat([z1, z2], dim=0)  # [2B, H]
        targets = torch.cat([target_items, target_items], dim=0)  # [2B]
        
        # Similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]
        
        # Mask for same-target pairs (positive pairs)
        target_mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
        
        # Remove self-similarity
        eye_mask = torch.eye(2 * batch_size, device=device)
        target_mask = target_mask - eye_mask
        
        # Check if there are any positive pairs
        if target_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # SupCon loss
        exp_sim = torch.exp(sim_matrix) * (1 - eye_mask)
        
        # For each anchor, sum over positive pairs
        pos_sim = exp_sim * target_mask
        neg_sim = exp_sim.sum(dim=1, keepdim=True)
        
        # Loss: -log(sum_pos / sum_all)
        loss = -torch.log(pos_sim.sum(dim=1) / (neg_sim.squeeze() + 1e-8))
        
        # Only compute for anchors with positive pairs
        valid_mask = target_mask.sum(dim=1) > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        loss = loss[valid_mask].mean()
        
        return loss
    
    def forward(
        self,
        repr_1: torch.Tensor,
        repr_2: torch.Tensor,
        target_items: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DuoRec loss.
        
        Args:
            repr_1: First view representation
            repr_2: Second view representation
            target_items: Target items for supervised loss
            
        Returns:
            (unsup_loss, sup_loss) tuple
        """
        unsup_loss = self.unsupervised_contrastive_loss(repr_1, repr_2)
        
        if target_items is not None:
            sup_loss = self.supervised_contrastive_loss(repr_1, repr_2, target_items)
        else:
            sup_loss = torch.tensor(0.0, device=repr_1.device)
        
        return unsup_loss, sup_loss
    
    def compute_total_loss(
        self,
        rec_loss: torch.Tensor,
        repr_1: torch.Tensor,
        repr_2: torch.Tensor,
        target_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total loss including recommendation and DuoRec components.
        """
        unsup_loss, sup_loss = self.forward(repr_1, repr_2, target_items)
        
        total_loss = (
            rec_loss +
            self.unsup_weight * unsup_loss +
            self.sup_weight * sup_loss
        )
        
        return total_loss
