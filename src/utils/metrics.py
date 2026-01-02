"""
Evaluation metrics for sequential recommendation.
"""

import numpy as np
import torch
from typing import Dict, List, Union


def NDCG(rank: int, k: int = 10) -> float:
    """
    Compute NDCG@K for a single sample.
    
    Args:
        rank: 0-indexed rank of the ground truth item
        k: cutoff
        
    Returns:
        NDCG score
    """
    if rank < k:
        return 1.0 / np.log2(rank + 2)
    return 0.0


def HitRate(rank: int, k: int = 10) -> float:
    """
    Compute Hit Rate@K for a single sample.
    
    Args:
        rank: 0-indexed rank of the ground truth item
        k: cutoff
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    return 1.0 if rank < k else 0.0


def MRR(rank: int, k: int = 10) -> float:
    """
    Compute MRR@K for a single sample.
    
    Args:
        rank: 0-indexed rank of the ground truth item
        k: cutoff
        
    Returns:
        Reciprocal rank
    """
    if rank < k:
        return 1.0 / (rank + 1)
    return 0.0


def compute_rank(
    scores: torch.Tensor,
    target_idx: int = 0,
) -> int:
    """
    Compute rank of target item.
    
    Args:
        scores: [num_candidates] prediction scores
        target_idx: index of target item in scores (default: 0, first item)
        
    Returns:
        0-indexed rank
    """
    # Sort scores in descending order
    sorted_indices = scores.argsort(descending=True)
    
    # Find rank of target
    rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item()
    
    return rank


def compute_metrics(
    scores: torch.Tensor,
    target_idx: int = 0,
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Compute all metrics for a single sample.
    
    Args:
        scores: [num_candidates] prediction scores
        target_idx: index of target item
        ks: list of cutoffs
        
    Returns:
        Dictionary of metrics
    """
    rank = compute_rank(scores, target_idx)
    
    metrics = {}
    for k in ks:
        metrics[f"ndcg@{k}"] = NDCG(rank, k)
        metrics[f"hr@{k}"] = HitRate(rank, k)
        metrics[f"mrr@{k}"] = MRR(rank, k)
    
    return metrics


def compute_batch_metrics(
    scores: torch.Tensor,
    target_indices: torch.Tensor = None,
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Compute average metrics for a batch.
    
    Args:
        scores: [batch_size, num_candidates] prediction scores
        target_indices: [batch_size] indices of target items (default: 0 for all)
        ks: list of cutoffs
        
    Returns:
        Dictionary of averaged metrics
    """
    batch_size = scores.shape[0]
    
    if target_indices is None:
        target_indices = torch.zeros(batch_size, dtype=torch.long)
    
    # Initialize accumulators
    metric_sums = {f"{m}@{k}": 0.0 for k in ks for m in ["ndcg", "hr", "mrr"]}
    
    for i in range(batch_size):
        sample_metrics = compute_metrics(scores[i], target_indices[i].item(), ks)
        for key, value in sample_metrics.items():
            metric_sums[key] += value
    
    # Average
    metrics = {key: value / batch_size for key, value in metric_sums.items()}
    
    return metrics


class MetricTracker:
    """
    Track and aggregate metrics during evaluation.
    """
    
    def __init__(self, ks: List[int] = [5, 10, 20]):
        self.ks = ks
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            f"{m}@{k}": [] for k in self.ks for m in ["ndcg", "hr", "mrr"]
        }
        self.count = 0
    
    def update(
        self,
        scores: torch.Tensor,
        target_idx: int = 0,
    ):
        """Update with a single sample."""
        sample_metrics = compute_metrics(scores, target_idx, self.ks)
        
        for key, value in sample_metrics.items():
            self.metrics[key].append(value)
        
        self.count += 1
    
    def update_batch(
        self,
        scores: torch.Tensor,
        target_indices: torch.Tensor = None,
    ):
        """Update with a batch."""
        batch_size = scores.shape[0]
        
        if target_indices is None:
            target_indices = torch.zeros(batch_size, dtype=torch.long)
        
        for i in range(batch_size):
            self.update(scores[i], target_indices[i].item())
    
    def compute(self) -> Dict[str, float]:
        """Compute final averaged metrics."""
        return {
            key: np.mean(values) if values else 0.0
            for key, values in self.metrics.items()
        }
    
    def __str__(self) -> str:
        metrics = self.compute()
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        return " | ".join(parts)


def evaluate_model(
    model,
    eval_loader,
    device: str = "cuda",
    num_negatives: int = 100,
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        eval_loader: Evaluation data loader
        device: Device to use
        num_negatives: Number of negative samples per positive
        ks: Cutoffs for metrics
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    tracker = MetricTracker(ks)
    
    with torch.no_grad():
        for batch in eval_loader:
            input_seq = batch["input_seq"].to(device)
            target_item = batch["target_item"].to(device)
            
            # Create candidate items: [target, negatives...]
            batch_size = input_seq.shape[0]
            
            # Get predictions
            # Assuming model.predict returns scores for candidates
            # where first item is target
            scores = model.predict(input_seq)
            
            # Get score for target and random negatives
            target_scores = scores.gather(1, target_item.unsqueeze(1))  # [B, 1]
            
            # Sample negative scores
            neg_indices = torch.randint(
                1, scores.shape[1], 
                (batch_size, num_negatives),
                device=device
            )
            neg_scores = scores.gather(1, neg_indices)  # [B, num_neg]
            
            # Combine: [target, neg1, neg2, ...]
            combined_scores = torch.cat([target_scores, neg_scores], dim=1)
            
            # Update tracker (target is at index 0)
            tracker.update_batch(combined_scores)
    
    return tracker.compute()
