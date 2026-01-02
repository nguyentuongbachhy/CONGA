"""
InfoNCE and contrastive loss implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss.
    
    Standard contrastive loss used in CL4SRec, SimCLR, etc.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: [batch_size, hidden_size] query representations
            positive_key: [batch_size, hidden_size] positive samples
            negative_keys: [batch_size, num_neg, hidden_size] or None
                          If None, uses in-batch negatives
                          
        Returns:
            loss: scalar
        """
        batch_size = query.shape[0]
        
        # Normalize
        if self.normalize:
            query = F.normalize(query, dim=-1)
            positive_key = F.normalize(positive_key, dim=-1)
        
        # Positive similarity
        pos_sim = (query * positive_key).sum(dim=-1) / self.temperature  # [B]
        
        if negative_keys is None:
            # In-batch negatives
            # All other samples in batch are negatives
            sim_matrix = torch.matmul(query, positive_key.T) / self.temperature  # [B, B]
            
            # Mask out positive pairs (diagonal)
            labels = torch.arange(batch_size, device=query.device)
            loss = F.cross_entropy(sim_matrix, labels)
        else:
            # Explicit negatives
            if self.normalize:
                negative_keys = F.normalize(negative_keys, dim=-1)
            
            neg_sim = torch.bmm(
                negative_keys, 
                query.unsqueeze(-1)
            ).squeeze(-1) / self.temperature  # [B, num_neg]
            
            # Concatenate positive and negatives
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+num_neg]
            
            # Labels: positive is always at index 0
            labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
            loss = F.cross_entropy(logits, labels)
        
        return loss


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    
    Symmetric version of InfoNCE, used in SimCLR.
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z1: [batch_size, hidden_size] first view
            z2: [batch_size, hidden_size] second view
            
        Returns:
            loss: scalar
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Concatenate
        z = torch.cat([z1, z2], dim=0)  # [2B, H]
        
        # Similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]
        
        # Mask self-similarity
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim.masked_fill_(mask, float('-inf'))
        
        # Positive pair indices
        # For i in [0, B-1], positive is at i + B
        # For i in [B, 2B-1], positive is at i - B
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device).bool()
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True
        
        # Compute loss
        pos_sim = sim[pos_mask].view(2 * batch_size, 1)
        neg_sim = sim[~mask & ~pos_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class HardNegativeInfoNCE(nn.Module):
    """
    InfoNCE with hard negative mining.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        hard_neg_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_weight = hard_neg_weight
    
    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE with hard negatives.
        
        Args:
            query: [batch_size, hidden_size]
            positive_key: [batch_size, hidden_size]
            hard_negatives: [batch_size, num_hard_neg, hidden_size]
            
        Returns:
            loss: scalar
        """
        batch_size = query.shape[0]
        device = query.device
        
        # Normalize
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        
        # In-batch similarity
        sim_matrix = torch.matmul(query, positive_key.T) / self.temperature
        
        # Add hard negatives if provided
        if hard_negatives is not None:
            hard_negatives = F.normalize(hard_negatives, dim=-1)
            hard_sim = torch.bmm(
                hard_negatives,
                query.unsqueeze(-1)
            ).squeeze(-1) / self.temperature  # [B, num_hard]
            
            # Weight hard negatives
            hard_sim = hard_sim * self.hard_neg_weight
            
            # Expand sim_matrix to include hard negatives
            sim_matrix = torch.cat([sim_matrix, hard_sim], dim=1)
        
        # Labels (positive is at diagonal position)
        labels = torch.arange(batch_size, device=device)
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
