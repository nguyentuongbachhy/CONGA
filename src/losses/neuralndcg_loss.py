"""
NeuralNDCG: Direct optimization of NDCG metric via differentiable approximation.

Reference: "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable 
           Relaxation of Sorting" (SIGIR eCom 2021)
           
Uses NeuralSort for differentiable sorting to create an arbitrary-accuracy 
approximation of NDCG@K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class NeuralSort(nn.Module):
    """
    Differentiable sorting network using softmax relaxation.
    
    Converts hard sorting into a soft permutation matrix that is differentiable.
    """
    
    def __init__(self, tau: float = 1.0):
        """
        Args:
            tau: Temperature parameter for softmax relaxation.
                 Lower tau = closer to hard sorting (sharper)
                 Higher tau = softer relaxation (smoother gradients)
        """
        super().__init__()
        self.tau = tau
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute soft permutation matrix via NeuralSort.
        
        Args:
            scores: [batch_size, num_items] - scores to sort
            
        Returns:
            P_hat: [batch_size, num_items, num_items] - soft permutation matrix
                   P_hat[b, i, j] ≈ 1 if item j is at position i after sorting
        """
        # NeuralSort (Grover et al., 2019) formulation.
        # Produces a soft permutation matrix approximating sorting in descending order.
        #
        # P_hat[i, j] = softmax( ((n + 1 - 2i) * s_j - sum_k |s_j - s_k|) / tau )
        # where i is 1..n (rank positions).
        batch_size, n = scores.shape

        # Pairwise absolute differences |s_j - s_k|
        # [B, n, n]
        pairwise_abs = torch.abs(scores.unsqueeze(2) - scores.unsqueeze(1))
        sum_pairwise_abs = pairwise_abs.sum(dim=-1)  # [B, n]

        # Rank scaling factors for positions i=1..n
        # For descending sort, larger scores should map to smaller i (top ranks).
        scaling = (n + 1 - 2 * torch.arange(1, n + 1, device=scores.device, dtype=scores.dtype))  # [n]
        scaling = scaling.view(1, n, 1)  # [1, n, 1]

        # Broadcast item scores over positions
        scores_b = scores.unsqueeze(1).expand(batch_size, n, n)  # [B, n(pos), n(items)]
        sum_abs_b = sum_pairwise_abs.unsqueeze(1).expand(batch_size, n, n)  # [B, n(pos), n(items)]

        P_logits = (scaling * scores_b - sum_abs_b) / self.tau
        P_hat = torch.softmax(P_logits, dim=-1)
        return P_hat


class NeuralNDCGLoss(nn.Module):
    """
    NeuralNDCG loss for direct NDCG@K optimization.
    
    Computes a differentiable approximation of NDCG@K using NeuralSort.
    Loss = -NDCG@K (we want to maximize NDCG, so minimize negative NDCG)
    """
    
    def __init__(
        self,
        k: int = 10,
        tau: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-10,
    ):
        """
        Args:
            k: Cutoff for NDCG@K (e.g., 10 for NDCG@10)
            tau: Temperature for NeuralSort (lower = sharper, higher = smoother)
            reduction: 'mean', 'sum', or 'none'
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.reduction = reduction
        self.eps = eps
        self.neural_sort = NeuralSort(tau=tau)
    
    def compute_dcg(
        self,
        relevance: torch.Tensor,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Discounted Cumulative Gain.
        
        DCG@K = Σ (2^rel_i - 1) / log2(i + 2) for i in [0, k)
        
        Args:
            relevance: [batch_size, num_items] - relevance scores (sorted)
            k: cutoff (uses self.k if None)
            
        Returns:
            dcg: [batch_size] - DCG scores
        """
        if k is None:
            k = self.k
        # Clamp k to the available list length
        k = min(k, relevance.size(1))
        
        # Take top-k
        relevance_k = relevance[:, :k]
        
        # Compute gains: 2^rel - 1
        gains = torch.pow(2.0, relevance_k) - 1.0
        
        # Compute discounts: 1 / log2(i + 2)
        positions = torch.arange(1, k + 1, dtype=torch.float32, device=relevance.device)
        discounts = 1.0 / torch.log2(positions + 1.0)
        
        # DCG = sum(gains * discounts)
        dcg = torch.sum(gains * discounts.unsqueeze(0), dim=-1)
        
        return dcg
    
    def compute_ndcg(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute differentiable NDCG@K using NeuralSort.
        
        Args:
            scores: [batch_size, num_items] - predicted scores
            labels: [batch_size, num_items] - relevance labels
            k: cutoff (uses self.k if None)
            
        Returns:
            ndcg: [batch_size] - NDCG@K scores
        """
        if k is None:
            k = self.k
        
        batch_size, num_items = scores.shape
        # Clamp k to the number of candidates
        k = min(k, num_items)
        
        # Get soft permutation matrix via NeuralSort
        # P_hat[b, i, j] ≈ 1 if item j is at position i
        P_hat = self.neural_sort(scores)
        
        # Apply soft permutation to labels to get "sorted" relevance
        # sorted_labels[b, i] = Σ_j P_hat[b, i, j] * labels[b, j]
        sorted_labels = torch.matmul(P_hat, labels.unsqueeze(-1)).squeeze(-1)
        
        # Compute DCG with soft-sorted labels
        dcg = self.compute_dcg(sorted_labels, k=k)
        
        # Compute ideal DCG (sort labels in descending order)
        ideal_sorted_labels, _ = torch.sort(labels, dim=-1, descending=True)
        idcg = self.compute_dcg(ideal_sorted_labels, k=k)
        
        # NDCG = DCG / IDCG
        ndcg = dcg / (idcg + self.eps)
        
        return ndcg
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute NeuralNDCG loss.
        
        Args:
            logits: [batch_size, num_items] - predicted scores
            labels: [batch_size, num_items] - relevance labels (0 or 1 for binary)
            mask: [batch_size, num_items] - optional mask for valid items
            
        Returns:
            loss: scalar - negative NDCG@K (to minimize)
        """
        # Apply mask if provided
        if mask is not None:
            # Set masked items to very low score and zero relevance
            logits = logits.masked_fill(~mask.bool(), -1e9)
            labels = labels.masked_fill(~mask.bool(), 0.0)
        
        # Compute NDCG@K
        ndcg = self.compute_ndcg(logits, labels, k=self.k)
        
        # Loss = -NDCG (we want to maximize NDCG)
        loss = -ndcg
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ApproxNDCGLoss(nn.Module):
    """
    Approximate NDCG loss using sigmoid smoothing (classical baseline).
    
    Simpler and faster than NeuralNDCG but less accurate approximation.
    Reference: "A General Approximation Framework for Direct Optimization of 
                Information Retrieval Measures" (ICML 2008)
    """
    
    def __init__(
        self,
        k: int = 10,
        alpha: float = 10.0,
        reduction: str = "mean",
        eps: float = 1e-10,
    ):
        """
        Args:
            k: Cutoff for NDCG@K
            alpha: Steepness of sigmoid approximation (higher = sharper)
            reduction: 'mean', 'sum', or 'none'
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute approximate NDCG loss.
        
        Uses sigmoid to approximate the ranking positions.
        """
        batch_size, num_items = logits.shape
        
        # Apply mask
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
            labels = labels.masked_fill(~mask.bool(), 0.0)
        
        # Compute pairwise score differences
        # [batch_size, num_items, num_items]
        score_diff = logits.unsqueeze(2) - logits.unsqueeze(1)
        
        # Approximate ranking with sigmoid
        # P(i ranked higher than j) ≈ sigmoid(alpha * (s_i - s_j))
        rank_prob = torch.sigmoid(self.alpha * score_diff)
        
        # Compute approximate rank for each item
        # rank[i] ≈ 1 + Σ_j P(j ranked higher than i)
        approx_rank = 1.0 + torch.sum(rank_prob, dim=1)
        
        # Compute gains and discounts
        gains = torch.pow(2.0, labels) - 1.0
        discounts = 1.0 / torch.log2(approx_rank + 1.0)
        
        # DCG with approximate ranks
        dcg = torch.sum(gains * discounts, dim=-1)
        
        # Ideal DCG
        sorted_labels, _ = torch.sort(labels, dim=-1, descending=True)
        ideal_gains = torch.pow(2.0, sorted_labels) - 1.0
        positions = torch.arange(1, num_items + 1, dtype=torch.float32, device=logits.device)
        ideal_discounts = 1.0 / torch.log2(positions + 1.0)
        idcg = torch.sum(ideal_gains * ideal_discounts, dim=-1)
        
        # NDCG
        ndcg = dcg / (idcg + self.eps)
        
        # Loss = -NDCG
        loss = -ndcg
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
