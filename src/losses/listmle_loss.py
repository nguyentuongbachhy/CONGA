"""
ListMLE (List Maximum Likelihood Estimation) loss for learning to rank.

Based on the Plackett-Luce model for listwise ranking.
Reference: "Listwise approach to learning to rank: theory and algorithm" (ICML 2008)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ListMLELoss(nn.Module):
    """
    ListMLE loss for listwise ranking optimization.
    
    Maximizes the likelihood of the correct permutation based on relevance scores.
    Uses the Plackett-Luce model: P(π | scores) = ∏ exp(s_π(i)) / Σ exp(s_π(j≥i))
    
    Loss = -log P(correct_permutation | scores)
    
    This loss directly optimizes for ranking metrics like NDCG@K.
    Complexity: O(n log n) where n is the number of items.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-10,
    ):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ListMLE loss.
        
        Args:
            logits: [batch_size, num_items] - predicted scores for items
            labels: [batch_size, num_items] - relevance labels (higher = more relevant)
                    For sequential rec: typically binary (1 for target, 0 for negatives)
                    or can be ranking positions
            mask: [batch_size, num_items] - optional mask for valid items
            
        Returns:
            loss: scalar or [batch_size] depending on reduction
        """
        batch_size, num_items = logits.shape
        
        # Sort items by labels in descending order (most relevant first)
        # This gives us the "correct" permutation
        sorted_indices = torch.argsort(labels, dim=-1, descending=True)
        
        # Gather logits in the sorted order
        sorted_logits = torch.gather(logits, dim=-1, index=sorted_indices)
        
        # Compute ListMLE loss using the Plackett-Luce model
        # For each position i, compute: log(exp(s_i) / sum(exp(s_j) for j >= i))
        # This is equivalent to: s_i - log_sum_exp(s_j for j >= i)
        
        # Create a mask for the cumulative sum from right to left
        # We need to compute log_sum_exp for each suffix
        max_logits = sorted_logits.max(dim=-1, keepdim=True)[0]
        exp_logits = torch.exp(sorted_logits - max_logits)
        
        # Cumulative sum from right to left
        cumsum_exp = torch.flip(
            torch.cumsum(torch.flip(exp_logits, dims=[-1]), dim=-1),
            dims=[-1]
        )
        
        # Compute log probabilities
        log_probs = sorted_logits - max_logits - torch.log(cumsum_exp + self.eps)
        
        # Sum log probabilities (product in probability space)
        # Exclude the last position as it has probability 1
        loss = -log_probs[:, :-1].sum(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            # Reorder mask according to sorted indices
            sorted_mask = torch.gather(mask, dim=-1, index=sorted_indices)
            # Apply mask to loss (only count valid items)
            loss = loss * sorted_mask[:, :-1].sum(dim=-1)
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ListMLELossSimplified(nn.Module):
    """
    Simplified ListMLE loss for sequential recommendation.
    
    Assumes binary labels: 1 for positive (target) item, 0 for negatives.
    Optimized for the common case in sequential recommendation where we have
    1 positive and multiple negatives.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-10,
    ):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ListMLE loss for positive vs negatives.
        
        Args:
            pos_logits: [batch_size] or [batch_size, seq_len] - scores for positive items
            neg_logits: [batch_size, num_neg] or [batch_size, seq_len, num_neg] - scores for negatives
            mask: optional mask for valid positions
            
        Returns:
            loss: scalar
        """
        # Ensure pos_logits has the same number of dimensions as neg_logits
        if pos_logits.dim() < neg_logits.dim():
            pos_logits = pos_logits.unsqueeze(-1)
        
        # Concatenate positive and negative logits
        # [batch_size, 1 + num_neg] or [batch_size, seq_len, 1 + num_neg]
        all_logits = torch.cat([pos_logits, neg_logits], dim=-1)
        
        # Compute log_sum_exp for normalization
        max_logits = all_logits.max(dim=-1, keepdim=True)[0]
        log_sum_exp = max_logits + torch.log(
            torch.sum(torch.exp(all_logits - max_logits), dim=-1, keepdim=True) + self.eps
        )
        
        # ListMLE loss: -log P(positive is ranked first)
        # = -(pos_logit - log_sum_exp(all_logits))
        loss = -(pos_logits - log_sum_exp).squeeze(-1)
        
        # Apply mask
        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + self.eps)
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
