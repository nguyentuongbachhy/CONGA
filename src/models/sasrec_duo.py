"""
SASRec with DuoRec Loss: Contrastive regularization for sequential recommendation.
Paper: "Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation" (WWW 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .sasrec import SASRec


class SASRecDuo(SASRec):
    """
    SASRec with DuoRec contrastive loss.
    
    DuoRec addresses the representation degeneration problem by:
    1. Model-level augmentation: dropout as augmentation
    2. Contrastive regularization: push similar sequences closer
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 64,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
        norm_first: bool = True,
        device: str = "cuda",
        # DuoRec specific
        contrastive_weight: float = 0.1,
        temperature: float = 1.0,
        supervised_weight: float = 0.1,
    ):
        super().__init__(
            num_items=num_items,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            norm_first=norm_first,
            device=device,
        )
        
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.supervised_weight = supervised_weight
    
    def forward(
        self, 
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with DuoRec contrastive loss components.
        """
        batch_size = item_seq.shape[0]
        
        # First forward pass
        seq_output_1 = self.encode_sequence(item_seq)  # [B, L, H]
        
        # Second forward pass (different dropout mask = model-level augmentation)
        seq_output_2 = self.encode_sequence(item_seq)  # [B, L, H]
        
        # Get final representations
        final_repr_1 = seq_output_1[:, -1, :]  # [B, H]
        final_repr_2 = seq_output_2[:, -1, :]  # [B, H]
        
        # Standard recommendation loss components
        pos_emb = self.item_embedding(pos_items)  # [B, L, H]
        neg_emb = self.item_embedding(neg_items)  # [B, L, H]
        
        pos_logits = (seq_output_1 * pos_emb).sum(dim=-1)  # [B, L]
        neg_logits = (seq_output_1 * neg_emb).sum(dim=-1)  # [B, L]
        
        # Unsupervised contrastive loss (InfoNCE between two views)
        # Normalize representations
        repr_1_norm = F.normalize(final_repr_1, dim=-1)
        repr_2_norm = F.normalize(final_repr_2, dim=-1)
        
        # Similarity matrix
        sim_matrix = torch.matmul(repr_1_norm, repr_2_norm.T) / self.temperature  # [B, B]
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=item_seq.device)
        
        # Unsupervised contrastive loss (symmetric)
        unsup_cl_loss = (
            F.cross_entropy(sim_matrix, labels) + 
            F.cross_entropy(sim_matrix.T, labels)
        ) / 2
        
        # Supervised contrastive loss (sequences with same next item are positive)
        last_items = pos_items[:, -1]  # [B]
        
        # Create mask for same-target pairs
        target_mask = (last_items.unsqueeze(0) == last_items.unsqueeze(1)).float()  # [B, B]
        target_mask.fill_diagonal_(0)  # Exclude self
        
        # Supervised contrastive (if there are positive pairs)
        if target_mask.sum() > 0:
            # Compute supervised contrastive loss
            exp_sim = torch.exp(sim_matrix)
            
            # For each anchor, compute loss over positive pairs
            pos_sim = exp_sim * target_mask
            neg_sim = exp_sim.sum(dim=1, keepdim=True) - exp_sim.diag().unsqueeze(1)
            
            # Avoid division by zero
            sup_cl_loss = -torch.log(
                (pos_sim.sum(dim=1) + 1e-8) / (neg_sim.squeeze() + 1e-8)
            )
            sup_cl_loss = sup_cl_loss[target_mask.sum(dim=1) > 0].mean()
            
            if torch.isnan(sup_cl_loss):
                sup_cl_loss = torch.tensor(0.0, device=item_seq.device)
        else:
            sup_cl_loss = torch.tensor(0.0, device=item_seq.device)
        
        return {
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
            "seq_output": seq_output_1,
            "unsup_cl_loss": unsup_cl_loss,
            "sup_cl_loss": sup_cl_loss,
            "final_repr_1": final_repr_1,
            "final_repr_2": final_repr_2,
        }
    
    def compute_total_loss(
        self, 
        outputs: Dict[str, torch.Tensor],
        pos_items: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Compute total loss including DuoRec components.
        
        Args:
            outputs: forward pass outputs
            pos_items: positive items for masking padding
            criterion: base loss function (BCE)
            
        Returns:
            total_loss: combined loss
        """
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        
        # Mask for non-padding positions
        mask = (pos_items != 0).float()
        
        # Base recommendation loss
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        rec_loss = (
            criterion(pos_logits, pos_labels) * mask + 
            criterion(neg_logits, neg_labels) * mask
        ).sum() / mask.sum()
        
        # DuoRec contrastive losses
        unsup_cl_loss = outputs["unsup_cl_loss"]
        sup_cl_loss = outputs["sup_cl_loss"]
        
        # Total loss
        total_loss = (
            rec_loss + 
            self.contrastive_weight * unsup_cl_loss +
            self.supervised_weight * sup_cl_loss
        )
        
        return total_loss
