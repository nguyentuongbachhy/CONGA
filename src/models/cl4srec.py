"""
CL4SRec: Contrastive Learning for Sequential Recommendation
Paper: https://arxiv.org/abs/2010.14395 (SIGIR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import random

from .sasrec import SASRec


class SequenceAugmentor:
    """
    Data augmentation for sequences.
    
    Three augmentation types from CL4SRec:
    1. Item Crop: randomly crop a contiguous subsequence
    2. Item Mask: randomly mask items with [mask] token
    3. Item Reorder: randomly reorder a contiguous subsequence
    """
    
    def __init__(
        self,
        crop_ratio: float = 0.6,
        mask_ratio: float = 0.3,
        reorder_ratio: float = 0.6,
        mask_token: int = 0,  # Use padding token as mask
    ):
        self.crop_ratio = crop_ratio
        self.mask_ratio = mask_ratio
        self.reorder_ratio = reorder_ratio
        self.mask_token = mask_token
    
    def crop(self, seq: torch.Tensor) -> torch.Tensor:
        """Randomly crop a contiguous subsequence."""
        batch_size, seq_len = seq.shape
        device = seq.device
        
        # Determine crop length for each sequence
        lengths = (seq != 0).sum(dim=1).float()
        crop_lens = (lengths * self.crop_ratio).long().clamp(min=1)
        
        cropped = torch.zeros_like(seq)
        
        for i in range(batch_size):
            valid_len = int(lengths[i].item())
            if valid_len <= 1:
                cropped[i] = seq[i]
                continue
            
            crop_len = min(int(crop_lens[i].item()), valid_len)
            start_idx = random.randint(0, valid_len - crop_len)
            
            # Find where valid items start in the sequence
            first_valid = (seq[i] != 0).nonzero(as_tuple=True)[0]
            if len(first_valid) == 0:
                cropped[i] = seq[i]
                continue
            
            first_valid_idx = first_valid[0].item()
            
            # Extract crop and right-align
            crop_start = first_valid_idx + start_idx
            crop_end = crop_start + crop_len
            cropped[i, seq_len - crop_len:] = seq[i, crop_start:crop_end]
        
        return cropped
    
    def mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Randomly mask items."""
        batch_size, seq_len = seq.shape
        device = seq.device
        
        masked = seq.clone()
        
        for i in range(batch_size):
            valid_indices = (seq[i] != 0).nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                continue
            
            num_mask = max(1, int(len(valid_indices) * self.mask_ratio))
            mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_mask]]
            masked[i, mask_indices] = self.mask_token
        
        return masked
    
    def reorder(self, seq: torch.Tensor) -> torch.Tensor:
        """Randomly reorder a contiguous subsequence."""
        batch_size, seq_len = seq.shape
        device = seq.device
        
        reordered = seq.clone()
        
        for i in range(batch_size):
            valid_indices = (seq[i] != 0).nonzero(as_tuple=True)[0]
            if len(valid_indices) <= 1:
                continue
            
            valid_len = len(valid_indices)
            reorder_len = max(2, int(valid_len * self.reorder_ratio))
            start_idx = random.randint(0, valid_len - reorder_len)
            
            # Get indices to reorder
            reorder_indices = valid_indices[start_idx:start_idx + reorder_len]
            
            # Shuffle the items at these positions
            items = reordered[i, reorder_indices].clone()
            perm = torch.randperm(len(items))
            reordered[i, reorder_indices] = items[perm]
        
        return reordered
    
    def augment(
        self, 
        seq: torch.Tensor, 
        aug_types: List[str] = ["crop", "mask", "reorder"]
    ) -> torch.Tensor:
        """Apply random augmentation."""
        aug_type = random.choice(aug_types)
        
        if aug_type == "crop":
            return self.crop(seq)
        elif aug_type == "mask":
            return self.mask(seq)
        elif aug_type == "reorder":
            return self.reorder(seq)
        else:
            return seq


class CL4SRec(SASRec):
    """
    CL4SRec: Contrastive Learning for Sequential Recommendation.
    
    Extends SASRec with:
    1. Data augmentation (crop, mask, reorder)
    2. Contrastive learning objective
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
        # CL4SRec specific
        contrastive_weight: float = 0.1,
        temperature: float = 1.0,
        augmentation_types: List[str] = ["crop", "mask", "reorder"],
        crop_ratio: float = 0.6,
        mask_ratio: float = 0.3,
        reorder_ratio: float = 0.6,
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
        self.augmentation_types = augmentation_types
        
        # Augmentor
        self.augmentor = SequenceAugmentor(
            crop_ratio=crop_ratio,
            mask_ratio=mask_ratio,
            reorder_ratio=reorder_ratio,
        )
    
    def forward(
        self, 
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with contrastive learning.
        """
        batch_size = item_seq.shape[0]
        
        # Standard forward pass
        seq_output = self.encode_sequence(item_seq)  # [B, L, H]
        
        # Get item embeddings
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        
        pos_logits = (seq_output * pos_emb).sum(dim=-1)
        neg_logits = (seq_output * neg_emb).sum(dim=-1)
        
        # Generate two augmented views
        aug_seq_1 = self.augmentor.augment(item_seq, self.augmentation_types)
        aug_seq_2 = self.augmentor.augment(item_seq, self.augmentation_types)
        
        # Encode augmented sequences
        aug_output_1 = self.encode_sequence(aug_seq_1)
        aug_output_2 = self.encode_sequence(aug_seq_2)
        
        # Get final representations
        aug_repr_1 = aug_output_1[:, -1, :]  # [B, H]
        aug_repr_2 = aug_output_2[:, -1, :]  # [B, H]
        
        # Normalize
        aug_repr_1 = F.normalize(aug_repr_1, dim=-1)
        aug_repr_2 = F.normalize(aug_repr_2, dim=-1)
        
        # Contrastive loss (InfoNCE)
        sim_matrix = torch.matmul(aug_repr_1, aug_repr_2.T) / self.temperature
        labels = torch.arange(batch_size, device=item_seq.device)
        
        cl_loss = (
            F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)
        ) / 2
        
        return {
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
            "seq_output": seq_output,
            "cl_loss": cl_loss,
            "aug_repr_1": aug_repr_1,
            "aug_repr_2": aug_repr_2,
        }
    
    def compute_total_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        pos_items: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute total loss including contrastive component."""
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        cl_loss = outputs["cl_loss"]
        
        # Mask for non-padding positions
        mask = (pos_items != 0).float()
        
        # Base recommendation loss
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        rec_loss = (
            criterion(pos_logits, pos_labels) * mask +
            criterion(neg_logits, neg_labels) * mask
        ).sum() / mask.sum()
        
        # Total loss
        total_loss = rec_loss + self.contrastive_weight * cl_loss
        
        return total_loss
