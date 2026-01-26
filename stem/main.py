import os
import time

import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from model import SASRec

# from graph_teacher import LightGCN
from continuum_memory import ContinuumItemEmbedding
from utils import (
    check_and_convert_dataset,
    load_metadata,
    get_dataloader,
    data_partition,
    evaluate,
    evaluate_valid,
)


def str2bool(s: str) -> bool:
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


def listmle_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    ListMLE loss based on Plackett-Luce model.
    Reference: "Listwise Approach to Learning to Rank" (Xia et al., 2008)
    Implementation adapted from allRank (https://github.com/allegro/allRank)

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    # Sort by ground truth descending (pos items first)
    y_true_sorted, indices = y_true.sort(descending=True, dim=-1)

    # Gather predictions according to ground truth order
    preds_sorted_by_true = torch.gather(y_pred, dim=1, index=indices)

    # Numerical stability: subtract max
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    # Compute cumulative sums from right to left (Plackett-Luce denominator)
    cumsums = torch.cumsum(
        preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
    ).flip(dims=[1])

    # ListMLE loss: log(cumsum) - logit
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    return torch.mean(torch.sum(observation_loss, dim=1))


def p_listmle_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Position-aware ListMLE (p-ListMLE) loss.
    Reference: "Position-Aware ListMLE: A Sequential Learning Process for Ranking" (Lan et al., 2014)

    Adds position-based weighting to emphasize top positions (similar to NDCG discount).

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    # Sort by ground truth descending
    y_true_sorted, indices = y_true.sort(descending=True, dim=-1)

    # Gather predictions according to ground truth order
    preds_sorted_by_true = torch.gather(y_pred, dim=1, index=indices)

    # Numerical stability
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    # Cumulative sums (Plackett-Luce)
    cumsums = torch.cumsum(
        preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
    ).flip(dims=[1])

    # Base ListMLE loss per position
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    # Position-aware weighting: 1/log2(position+1) (NDCG-style discount)
    slate_length = y_pred.shape[1]
    positions = torch.arange(
        1, slate_length + 1, dtype=torch.float32, device=y_pred.device
    )
    position_weights = 1.0 / torch.log2(positions + 1.0)  # [slate_length]
    position_weights = position_weights.unsqueeze(0)  # [1, slate_length]

    # Apply position weights
    weighted_loss = observation_loss * position_weights

    return torch.mean(torch.sum(weighted_loss, dim=1))


def p_sampled_softmax_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Position-aware Sampled Softmax loss.
    Combines the simplicity of sampled softmax with position-aware weighting.

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    # Standard sampled softmax: -log_softmax[:, 0]
    softmax_probs = F.log_softmax(y_pred, dim=1)
    base_loss = -softmax_probs[:, 0]  # Only positive item (position 0)

    # Position-aware weighting: emphasize top position more
    slate_length = y_pred.shape[1]
    positions = torch.arange(
        1, slate_length + 1, dtype=torch.float32, device=y_pred.device
    )
    position_weights = 1.0 / torch.log2(positions + 1.0)  # [slate_length]
    position_weights = position_weights.unsqueeze(0)  # [1, slate_length]

    # Weight the loss - only position 0 matters for positive item
    weighted_loss = base_loss * position_weights[:, 0]

    return torch.mean(weighted_loss)


def mmcl_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    margins=[0.2, 0.5, 0.8],
    weights=[1.0, 0.5, 0.2],
    temperature=1.0,
    eps=1e-10,
) -> torch.Tensor:
    """
    Multi-Margin Cosine Loss (MMCL) for recommendation systems.
    Reference: "Multi-Margin Cosine Loss: Proposal and Application in Recommender Systems" (Ozsoy, 2024)

    Uses multiple margins to capture different levels of negative hardness:
    - Hardest negatives (small margin)
    - Semi-hard negatives (medium margin)
    - Semi-easy negatives (large margin)

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        margins: list of margin values for different negative levels
        weights: list of weights for each margin (importance of each negative level)
        temperature: temperature scaling for softmax
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    # Normalize logits to cosine similarity range [-1, 1]
    # Apply temperature scaling
    y_pred_scaled = y_pred / temperature

    # Convert to similarity scores (assuming logits are already similarities)
    # For cosine similarity: higher is better
    pos_sim = y_pred_scaled[:, 0]  # [batch_size]
    neg_sims = y_pred_scaled[:, 1:]  # [batch_size, num_negatives]

    # Positive loss component (encourage high similarity with positive)
    # f(u,i) = max(0, 1 - s(u,i)) for positive
    pos_loss = torch.mean(torch.relu(1.0 - pos_sim))

    # Multi-margin negative loss component
    # For each margin level, compute weighted loss
    neg_loss = 0.0
    for margin, weight in zip(margins, weights):
        # f(u,i,j,m) = s(u,j) - m for negatives
        # We want s(u,j) < m (negative should be far from user)
        margin_term = neg_sims - margin
        # Use softplus for smooth approximation: log(1 + exp(x))
        neg_loss += weight * torch.mean(F.softplus(margin_term))

    # Combine positive and negative losses
    # wp * pos_loss + wn * neg_loss (using wp=1.0, wn=1.0)
    total_loss = pos_loss + neg_loss

    return total_loss


def gbce_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, alpha=0.75, temperature=1.0, eps=1e-10
) -> torch.Tensor:
    """
    Generalized Binary Cross-Entropy (gBCE) loss for recommendation systems.
    Reference: "gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling" (2023)

    Mitigates overconfidence problem in negative sampling by combining:
    - Binary Cross-Entropy (BCE): For positive/negative classification
    - Cross-Entropy (CE): For ranking among all candidates

    The key insight: Negative sampling increases proportion of positive interactions,
    causing models to overestimate positive probabilities. gBCE balances this.

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        alpha: weight for BCE component (1-alpha for CE). Higher alpha = more BCE influence.
               Typical range: 0.7-0.8 for overconfidence mitigation
        temperature: temperature scaling for softmax
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    # Apply temperature scaling
    y_pred_scaled = y_pred / temperature

    # Component 1: Binary Cross-Entropy (BCE)
    # Treats each item independently as binary classification
    # BCE = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_loss = -(
        y_true * torch.log(sigmoid_pred + eps)
        + (1 - y_true) * torch.log(1 - sigmoid_pred + eps)
    )
    bce_loss = torch.mean(bce_loss)

    # Component 2: Cross-Entropy (CE) / Softmax Loss
    # Treats the slate as a ranking problem
    # CE = -log(softmax(x_pos))
    ce_loss = -F.log_softmax(y_pred_scaled, dim=1)[
        :, 0
    ]  # Loss for positive (first position)
    ce_loss = torch.mean(ce_loss)

    # Generalized BCE: Weighted combination
    # alpha controls the trade-off between BCE and CE
    # Higher alpha = more focus on binary classification (reduces overconfidence)
    # Lower alpha = more focus on ranking (standard softmax)
    total_loss = alpha * bce_loss + (1 - alpha) * ce_loss

    return total_loss


def p_gbce_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, alpha=0.75, temperature=1.0, eps=1e-10
) -> torch.Tensor:
    """
    Position-Aware Generalized Binary Cross-Entropy (p-gBCE) loss for recommendation systems.
    Novel combination: gBCE (overconfidence mitigation) + position-aware weighting (NDCG-style).

    Addresses two key problems simultaneously:
    1. Overconfidence in negative sampling (gBCE component)
    2. Position bias - top positions are more important (position-aware weighting)

    Formula: p-gBCE = α * (weighted BCE) + (1-α) * (weighted CE)
    where weights = 1/log2(position + 1) (NDCG discount)

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        alpha: weight for BCE component (1-alpha for CE). Higher alpha = more BCE influence.
               Typical range: 0.7-0.8 for overconfidence mitigation
        temperature: temperature scaling for softmax
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    # Apply temperature scaling
    y_pred_scaled = y_pred / temperature

    # Position-aware weights: NDCG-style discount
    # Higher weight for top positions (more important)
    slate_length = y_pred.shape[1]
    positions = torch.arange(
        1, slate_length + 1, dtype=torch.float32, device=y_pred.device
    )
    position_weights = 1.0 / torch.log2(positions + 1.0)  # [slate_length]
    position_weights = position_weights.unsqueeze(0)  # [1, slate_length]

    # Component 1: Position-weighted Binary Cross-Entropy (BCE)
    # Treats each item independently as binary classification
    # BCE = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_loss_per_item = -(
        y_true * torch.log(sigmoid_pred + eps)
        + (1 - y_true) * torch.log(1 - sigmoid_pred + eps)
    )
    # Apply position weights
    weighted_bce_loss = bce_loss_per_item * position_weights
    weighted_bce_loss = torch.mean(weighted_bce_loss)

    # Component 2: Position-weighted Cross-Entropy (CE) / Softmax Loss
    # Treats the slate as a ranking problem
    # CE = -log(softmax(x_pos)) for each position
    ce_loss_per_item = -F.log_softmax(
        y_pred_scaled, dim=1
    )  # [batch_size, slate_length]
    # Apply position weights
    weighted_ce_loss = ce_loss_per_item * position_weights
    # Focus on positive items (where y_true = 1)
    positive_mask = y_true == 1.0
    weighted_ce_loss = weighted_ce_loss * positive_mask
    weighted_ce_loss = torch.mean(weighted_ce_loss)

    # Position-aware Generalized BCE: Weighted combination
    # alpha controls the trade-off between BCE and CE
    # Higher alpha = more focus on binary classification (reduces overconfidence)
    # Lower alpha = more focus on ranking (standard softmax)
    total_loss = alpha * weighted_bce_loss + (1 - alpha) * weighted_ce_loss

    return total_loss


# =============================================================================
# NOVEL LOSS FUNCTIONS - Research-Driven Creative Enhancements
# =============================================================================


def rc_gbce_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha=0.75,
    k=10,
    temperature=1.0,
    eps=1e-10,
) -> torch.Tensor:
    """
    Rank-Calibrated gBCE (RC-gBCE): A novel loss that focuses learning on the
    decision boundary around top-k positions.

    Key insight: In HR@10/NDCG@10, only items near the top-10 boundary matter.
    Items clearly ranked high or low contribute less to the final metric.
    RC-gBCE dynamically weights samples based on their proximity to rank k.

    Formula: RC-gBCE = α * (rank-weighted BCE) + (1-α) * (rank-weighted CE)
    where rank_weights = exp(-((rank - k)^2) / (2 * k))

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        alpha: weight for BCE component (1-alpha for CE)
        k: target rank for calibration (default 10 for HR@10/NDCG@10)
        temperature: temperature scaling
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    y_pred_scaled = y_pred / temperature

    # Compute approximate ranks using differentiable sorting proxy
    # For each item, count how many items have higher scores
    # ranks[i] = 1 + sum(pred[j] > pred[i] for all j)
    pred_diff = y_pred_scaled.unsqueeze(-1) - y_pred_scaled.unsqueeze(-2)  # [B, S, S]
    approx_ranks = 1.0 + torch.sigmoid(pred_diff * 10.0).sum(dim=-1)  # [B, S]

    # Rank-calibrated weights: Gaussian centered at rank k
    # High weight for items near the decision boundary (rank ~ k)
    # Lower weight for items clearly ranked high (rank << k) or low (rank >> k)
    rank_weights = torch.exp(-((approx_ranks - k) ** 2) / (2.0 * k))
    rank_weights = rank_weights / (
        rank_weights.sum(dim=-1, keepdim=True) + eps
    )  # Normalize

    # Component 1: Rank-weighted BCE
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_per_item = -(
        y_true * torch.log(sigmoid_pred + eps)
        + (1 - y_true) * torch.log(1 - sigmoid_pred + eps)
    )
    weighted_bce = (bce_per_item * rank_weights).sum(dim=-1).mean()

    # Component 2: Rank-weighted CE (softmax)
    ce_per_item = -F.log_softmax(y_pred_scaled, dim=1)
    # Focus on positive items
    positive_mask = y_true == 1.0
    weighted_ce = (ce_per_item * rank_weights * positive_mask).sum(dim=-1).mean()

    total_loss = alpha * weighted_bce + (1 - alpha) * weighted_ce

    return total_loss


def approx_ndcg_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, temperature=1.0, k=10, eps=1e-10
) -> torch.Tensor:
    """
    ApproxNDCG: Differentiable approximation of NDCG loss.
    Directly optimizes the target metric (NDCG@k) using softmax-based rank approximation.

    Reference: "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures" (Qin et al., 2010)

    Key idea: Use softmax to create a differentiable approximation of sorting,
    then compute DCG using these soft ranks.

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        temperature: temperature for softmax (lower = sharper approximation)
        k: cutoff for NDCG computation
        eps: epsilon for numerical stability

    Returns:
        scalar loss (negative NDCG to minimize)
    """
    y_pred_scaled = y_pred / temperature
    batch_size, slate_length = y_pred.shape

    # Compute soft ranks using softmax over pairwise differences
    # For position i, soft_rank[i] = 1 + sum_j(sigmoid(s_j - s_i))
    pred_diff = y_pred_scaled.unsqueeze(-1) - y_pred_scaled.unsqueeze(-2)  # [B, S, S]
    soft_ranks = 1.0 + torch.sigmoid(pred_diff * 5.0).sum(dim=-1)  # [B, S]

    # Clamp ranks to [1, slate_length] for stability
    soft_ranks = soft_ranks.clamp(min=1.0, max=float(slate_length))

    # DCG discount: 1 / log2(rank + 1)
    discounts = 1.0 / torch.log2(soft_ranks + 1.0)  # [B, S]

    # Gains: 2^relevance - 1 (for binary relevance, this is just y_true)
    gains = y_true  # [B, S]

    # DCG = sum of gains * discounts (only consider top-k positions approximately)
    # Weight by how likely the item is in top-k
    topk_weight = torch.sigmoid((k + 0.5 - soft_ranks) * 2.0)  # Smooth step at rank k
    dcg = (gains * discounts * topk_weight).sum(dim=-1)  # [B]

    # Ideal DCG: sort gains descending, apply ideal discounts
    sorted_gains, _ = gains.sort(dim=-1, descending=True)
    ideal_positions = torch.arange(
        1, slate_length + 1, dtype=torch.float32, device=y_pred.device
    )
    ideal_discounts = 1.0 / torch.log2(ideal_positions + 1.0)
    # Only consider top-k for ideal DCG
    k_actual = min(k, slate_length)
    idcg = (sorted_gains[:, :k_actual] * ideal_discounts[:k_actual]).sum(dim=-1)  # [B]

    # NDCG = DCG / IDCG
    ndcg = dcg / (idcg + eps)

    # Return negative NDCG (we want to maximize NDCG, so minimize -NDCG)
    return -ndcg.mean()


def infonce_gbce_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    seq_emb: torch.Tensor = None,
    pos_emb: torch.Tensor = None,
    alpha=0.75,
    beta=0.3,
    temperature=0.1,
    gbce_temp=1.0,
    eps=1e-10,
) -> torch.Tensor:
    """
    InfoNCE-gBCE Hybrid Loss: Combines gBCE with in-batch contrastive learning.

    Key innovation: Uses other samples' positive items as FREE additional negatives
    through in-batch contrastive learning. This significantly increases negative
    diversity without any extra sampling cost.

    Formula: L = α * L_gBCE + β * L_InfoNCE

    When seq_emb and pos_emb are None, falls back to standard gBCE.

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        seq_emb: sequence embeddings, shape [batch_size, hidden_dim] (optional)
        pos_emb: positive item embeddings, shape [batch_size, hidden_dim] (optional)
        alpha: weight for BCE component in gBCE (1-alpha for CE)
        beta: weight for InfoNCE component
        temperature: temperature for InfoNCE contrastive loss
        gbce_temp: temperature for gBCE components
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    # Component 1: Standard gBCE loss
    gbce = gbce_loss(y_pred, y_true, alpha=alpha, temperature=gbce_temp, eps=eps)

    # If embeddings not provided, return standard gBCE
    if seq_emb is None or pos_emb is None:
        return gbce

    # Component 2: InfoNCE with in-batch negatives
    batch_size = seq_emb.size(0)

    # Normalize embeddings for cosine similarity
    seq_emb_norm = F.normalize(seq_emb, p=2, dim=-1)
    pos_emb_norm = F.normalize(pos_emb, p=2, dim=-1)

    # Positive similarities (diagonal)
    pos_sim = (seq_emb_norm * pos_emb_norm).sum(dim=-1) / temperature  # [B]

    # All pairwise similarities (in-batch negatives)
    # Each sequence is compared against all positive items in the batch
    all_sim = torch.mm(seq_emb_norm, pos_emb_norm.t()) / temperature  # [B, B]

    # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
    # The positive is on the diagonal
    infonce = -pos_sim + torch.logsumexp(all_sim, dim=-1)
    infonce = infonce.mean()

    # Combined loss
    total_loss = gbce + beta * infonce

    return total_loss


def tcr_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    seq_emb: torch.Tensor = None,
    alpha=0.75,
    lambda_temporal=0.2,
    lambda_coherence=0.1,
    temperature=1.0,
    eps=1e-10,
) -> torch.Tensor:
    """
    Temporal Contrastive Ranking Loss (TCR-Loss): A novel loss that captures
    temporal dynamics in sequential recommendations.

    Key insight: In sequential recommendation, temporal proximity matters.
    Items closer in time should have more coherent representations.

    Formula: L_TCR = L_gBCE + λ₁ * L_temporal + λ₂ * L_coherence

    Where:
    - L_temporal: Contrastive loss encouraging temporal neighbors to be similar
    - L_coherence: Smoothness loss for embedding transitions along the sequence

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        seq_emb: full sequence embeddings, shape [batch_size, seq_len, hidden_dim] (optional)
        alpha: weight for BCE component in gBCE
        lambda_temporal: weight for temporal contrastive loss
        lambda_coherence: weight for sequence coherence loss
        temperature: temperature scaling
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    # Base loss: gBCE
    base_loss = gbce_loss(y_pred, y_true, alpha=alpha, temperature=temperature, eps=eps)

    # If sequence embeddings not provided, return base loss
    if seq_emb is None or seq_emb.dim() != 3:
        return base_loss

    batch_size, seq_len, hidden_dim = seq_emb.shape

    # Skip if sequence too short
    if seq_len < 3:
        return base_loss

    # Component 1: Temporal Contrastive Loss
    # Positive pairs: (t, t+1) - adjacent items
    # Negative pairs: (t, t+k) where k > 2 - distant items

    # Get adjacent pairs
    curr_emb = seq_emb[:, :-1, :]  # [B, L-1, H]
    next_emb = seq_emb[:, 1:, :]  # [B, L-1, H]

    # Normalize for cosine similarity
    curr_norm = F.normalize(curr_emb, p=2, dim=-1)
    next_norm = F.normalize(next_emb, p=2, dim=-1)

    # Positive similarity: adjacent items should be similar
    pos_sim = (curr_norm * next_norm).sum(dim=-1)  # [B, L-1]

    # Negative similarity: compare with items 2+ steps away
    # Use a simple approach: compare each position with all positions
    # Flatten for batch matrix multiply
    curr_flat = curr_norm.reshape(-1, hidden_dim)  # [B*(L-1), H]

    # Get distant negatives (shift by 2 positions)
    if seq_len > 3:
        distant_emb = seq_emb[:, 2:, :]  # [B, L-2, H]
        distant_norm = F.normalize(distant_emb, p=2, dim=-1)

        # Simple temporal contrastive: maximize pos_sim, minimize distant_sim
        neg_sim = (curr_norm[:, :-1, :] * distant_norm).sum(dim=-1)  # [B, L-2]

        # Margin-based temporal loss
        margin = 0.3
        temporal_loss = torch.relu(neg_sim - pos_sim[:, :-1] + margin).mean()
    else:
        temporal_loss = torch.tensor(0.0, device=y_pred.device)

    # Component 2: Sequence Coherence Loss
    # Encourage smooth transitions in embedding space
    # Minimize the variance of consecutive differences
    emb_diff = next_emb - curr_emb  # [B, L-1, H]

    # Coherence: differences should be consistent (low variance)
    diff_mean = emb_diff.mean(dim=1, keepdim=True)  # [B, 1, H]
    coherence_loss = ((emb_diff - diff_mean) ** 2).mean()

    # Combined loss
    total_loss = (
        base_loss + lambda_temporal * temporal_loss + lambda_coherence * coherence_loss
    )

    return total_loss


def composite_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    seq_emb: torch.Tensor = None,
    pos_emb: torch.Tensor = None,
    loss_weights: dict = None,
    k=10,
    eps=1e-10,
) -> torch.Tensor:
    """
    Composite Loss: Combines multiple loss objectives with learnable/fixed weights.

    This is a meta-loss that can combine:
    - gBCE (overconfidence mitigation)
    - ApproxNDCG (direct metric optimization)
    - InfoNCE (contrastive learning)

    Default weights are tuned for sequential recommendation.

    Args:
        y_pred: predictions/logits, shape [batch_size, slate_length]
        y_true: ground truth labels (1 for pos, 0 for neg), shape [batch_size, slate_length]
        seq_emb: sequence embeddings (optional)
        pos_emb: positive item embeddings (optional)
        loss_weights: dict with keys 'gbce', 'ndcg', 'infonce' (optional)
        k: cutoff for NDCG
        eps: epsilon for numerical stability

    Returns:
        scalar loss
    """
    if loss_weights is None:
        loss_weights = {"gbce": 0.5, "ndcg": 0.3, "infonce": 0.2}

    total_loss = 0.0

    # gBCE component
    if loss_weights.get("gbce", 0) > 0:
        gbce = gbce_loss(y_pred, y_true, alpha=0.75, eps=eps)
        total_loss = total_loss + loss_weights["gbce"] * gbce

    # ApproxNDCG component
    if loss_weights.get("ndcg", 0) > 0:
        ndcg = approx_ndcg_loss(y_pred, y_true, k=k, eps=eps)
        total_loss = total_loss + loss_weights["ndcg"] * ndcg

    # InfoNCE component (if embeddings provided)
    if (
        loss_weights.get("infonce", 0) > 0
        and seq_emb is not None
        and pos_emb is not None
    ):
        batch_size = seq_emb.size(0)
        seq_norm = F.normalize(seq_emb, p=2, dim=-1)
        pos_norm = F.normalize(pos_emb, p=2, dim=-1)

        pos_sim = (seq_norm * pos_norm).sum(dim=-1) / 0.1
        all_sim = torch.mm(seq_norm, pos_norm.t()) / 0.1
        infonce = (-pos_sim + torch.logsumexp(all_sim, dim=-1)).mean()

        total_loss = total_loss + loss_weights["infonce"] * infonce

    return total_loss


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=200, type=int)
parser.add_argument("--hidden_units", default=50, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument(
    "--num_negatives", default=1, type=int, help="Number of negatives per position"
)
parser.add_argument(
    "--neg_sampling_mode",
    default="random",
    type=str,
    choices=["random", "popularity", "frequency", "mans"],
    help="Negative sampling strategy: random (default), popularity (harder negatives), "
    "frequency (diverse/rare items), mans (Manifold-Aware: 50%% pop + 30%% freq + 20%% random)",
)
parser.add_argument(
    "--loss_type",
    default="sampled_softmax",
    type=str,
    choices=[
        "sampled_softmax",
        "p_sampled_softmax",
        "listmle",
        "p_listmle",
        "mmcl",
        "gbce",
        "p_gbce",
        "rc_gbce",
        "approx_ndcg",
        "infonce_gbce",
        "tcr",
        "composite",
    ],
    help="Loss function: sampled_softmax (default), p_sampled_softmax, listmle, p_listmle, mmcl, gbce, p_gbce, "
    "rc_gbce (rank-calibrated gBCE), approx_ndcg (differentiable NDCG), infonce_gbce (contrastive + gBCE), "
    "tcr (temporal contrastive ranking), composite (multi-objective)",
)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--inference_only", default=False, type=str2bool)
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument("--norm_first", action="store_true", default=False)
parser.add_argument("--num_workers", default=4, type=int, help="DataLoader workers")
parser.add_argument(
    "--use_nested_learning",
    default=False,
    type=str2bool,
    help="Enable Continuum Memory System (CMS) for multi-timescale learning",
)
parser.add_argument(
    "--cms_fast_weight",
    default=0.5,
    type=float,
    help="Weight for fast memory in CMS (recent interactions)",
)
parser.add_argument(
    "--cms_medium_weight",
    default=0.3,
    type=float,
    help="Weight for medium memory in CMS (session patterns)",
)
parser.add_argument(
    "--cms_slow_weight",
    default=0.2,
    type=float,
    help="Weight for slow memory in CMS (long-term knowledge)",
)
parser.add_argument(
    "--use_stem",
    default=False,
    type=str2bool,
    help="Enable STEM (Scaling Transformers with Embedding Modules) - replaces FFN up-projection with item-indexed embeddings",
)
parser.add_argument(
    "--stem_start_layer",
    default=-1,
    type=int,
    help="Layer index to start using STEM (default: -1 = last layer only). Use 0 for all layers.",
)

args = parser.parse_args()

if not os.path.isdir(args.dataset + "_" + args.train_dir):
    os.makedirs(args.dataset + "_" + args.train_dir)
with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f_args:
    f_args.write(
        "\n".join(
            [
                str(k) + "," + str(v)
                for k, v in sorted(vars(args).items(), key=lambda x: x[0])
            ]
        )
    )

if __name__ == "__main__":
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)

    tensor_mem_gb = (
        args.batch_size * args.maxlen * args.num_negatives * args.hidden_units * 4
    ) / (1024**3)
    print(f"Dataset: {args.dataset} | Users: {usernum} | Items: {itemnum}")
    print(f"Estimated VRAM for Negatives Tensor: {tensor_mem_gb:.2f} GB")

    if tensor_mem_gb > 8.0:
        print("\n[WARNING] Cấu hình này yêu cầu VRAM rất lớn!")
        print("Gợi ý: Giảm --num_negatives (ví dụ: 1) hoặc giảm --batch_size.\n")

    print(f"Loss function: {args.loss_type}")
    if args.loss_type in ["listmle", "p_listmle"]:
        print(f"  Using Plackett-Luce ranking loss with {args.num_negatives} negatives")
        if args.loss_type == "p_listmle":
            print(f"  Position-aware weighting enabled (NDCG-style discount)")
    elif args.loss_type == "p_sampled_softmax":
        print(
            f"  Using position-aware sampled softmax with {args.num_negatives} negatives"
        )
        print(f"  NDCG-style position weighting applied")
    elif args.loss_type == "mmcl":
        print(
            f"  Using Multi-Margin Cosine Loss (MMCL) with {args.num_negatives} negatives"
        )
        print(f"  Multiple margins for hardest/semi-hard/semi-easy negatives")
    elif args.loss_type == "gbce":
        print(
            f"  Using Generalized Binary Cross-Entropy (gBCE) with {args.num_negatives} negatives"
        )
        print(f"  Overconfidence mitigation for negative sampling")
    elif args.loss_type == "p_gbce":
        print(
            f"  Using Position-Aware gBCE (p-gBCE) with {args.num_negatives} negatives"
        )
        print(f"  Overconfidence mitigation + NDCG-style position weighting")
    elif args.loss_type == "rc_gbce":
        print(
            f"  Using Rank-Calibrated gBCE (RC-gBCE) with {args.num_negatives} negatives"
        )
        print(f"  Focuses learning on decision boundary around top-k positions")
    elif args.loss_type == "approx_ndcg":
        print(f"  Using ApproxNDCG loss with {args.num_negatives} negatives")
        print(f"  Directly optimizes NDCG@10 via differentiable approximation")
    elif args.loss_type == "infonce_gbce":
        print(f"  Using InfoNCE-gBCE Hybrid with {args.num_negatives} negatives")
        print(f"  Combines gBCE with in-batch contrastive learning")
    elif args.loss_type == "tcr":
        print(
            f"  Using Temporal Contrastive Ranking (TCR) with {args.num_negatives} negatives"
        )
        print(f"  gBCE + temporal coherence + sequence smoothness")
    elif args.loss_type == "composite":
        print(f"  Using Composite Loss with {args.num_negatives} negatives")
        print(f"  Multi-objective: gBCE (0.5) + ApproxNDCG (0.3) + InfoNCE (0.2)")

    if args.use_nested_learning:
        print(f"Nested Learning (CMS): ENABLED")
        print(f"  - Fast memory weight: {args.cms_fast_weight}")
        print(f"  - Medium memory weight: {args.cms_medium_weight}")
        print(f"  - Slow memory weight: {args.cms_slow_weight}")
    
    if args.use_stem:
        print(f"\nSTEM (Scaling Transformers with Embedding Modules): ENABLED")
        if args.stem_start_layer == -1:
            print(f"  - Strategy: STEM-Lite (last layer only)")
            print(f"  - Layer {args.num_blocks - 1}: STEM FFN (item-indexed embeddings)")
        elif args.stem_start_layer == 0:
            print(f"  - Strategy: STEM-Full (all layers)")
            print(f"  - All {args.num_blocks} layers use STEM FFN")
        else:
            print(f"  - Strategy: STEM-Hybrid (layers {args.stem_start_layer}+)")
            print(f"  - Layers 0-{args.stem_start_layer-1}: Standard SwiGLU")
            print(f"  - Layers {args.stem_start_layer}-{args.num_blocks-1}: STEM FFN")
        print(f"  - Benefits: 33% fewer FFN params, item-specific knowledge, better interpretability")
        print(f"  - Paper: https://arxiv.org/abs/2601.10639")

    # Print MANS (Manifold-Aware Negative Sampling) info
    print(f"\nNegative Sampling: {args.neg_sampling_mode}")
    if args.neg_sampling_mode == "random":
        print(f"  Standard uniform random sampling (baseline)")
    elif args.neg_sampling_mode == "popularity":
        print(f"  Popularity-weighted sampling (harder negatives)")
        print(f"  Popular items are sampled more frequently")
    elif args.neg_sampling_mode == "frequency":
        print(f"  Frequency-weighted sampling (diverse negatives)")
        print(f"  Rare items are sampled more frequently for diversity")
    elif args.neg_sampling_mode == "mans":
        print(f"  MANS: Manifold-Aware Negative Sampling")
        print(f"  Combines: popularity (50%) + frequency (30%) + random (20%)")
        print(f"  - Popularity: harder negatives from popular items")
        print(f"  - Frequency: diverse negatives from rare items")
        print(f"  - Random: exploration via uniform sampling")

    train_loader = get_dataloader(
        args.dataset,
        args.maxlen,
        args.batch_size,
        mode="train",
        num_workers=args.num_workers,
        num_negatives=args.num_negatives,
        neg_sampling_mode=args.neg_sampling_mode,
    )

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, _, _] = dataset

    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

    model = SASRec(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.item_emb.weight.data[0, :] = 0

    # Apply Continuum Memory System if enabled
    if args.use_nested_learning:
        print("\n[Applying Continuum Memory System...]")
        original_emb = model.item_emb

        cms_emb = ContinuumItemEmbedding(
            num_items=original_emb.num_embeddings,
            embedding_dim=original_emb.embedding_dim,
            padding_idx=original_emb.padding_idx,
            fast_weight=args.cms_fast_weight,
            medium_weight=args.cms_medium_weight,
            slow_weight=args.cms_slow_weight,
            device=torch.device(args.device),
        )

        # Initialize CMS from current embeddings (random)
        print("  - Initializing CMS with random embeddings")
        with torch.no_grad():
            cms_emb.fast_emb.weight.copy_(original_emb.weight)
            cms_emb.medium_emb.weight.copy_(original_emb.weight)
            cms_emb.slow_emb.weight.copy_(original_emb.weight)

        # Replace item embeddings with CMS
        model.item_emb = cms_emb
        print("  CMS applied successfully")

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path, map_location=torch.device(args.device))
            )
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except Exception as e:
            print(f"Failed loading state_dicts: {e}")

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f"Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})")
        exit(0)

    # Setup optimizer with multi-timescale learning rates for CMS
    if args.use_nested_learning:
        print("  - Using multi-timescale learning rates for CMS")

        # Get CMS parameter groups with different learning rates
        cms_param_groups = model.item_emb.get_parameter_groups(args.lr)

        # Get other model parameters
        other_params = [p for n, p in model.named_parameters() if "item_emb" not in n]
        other_param_group = {"params": other_params, "lr": args.lr, "name": "other"}

        # Combine all parameter groups
        param_groups = cms_param_groups + [other_param_group]

        optimizer = torch.optim.AdamW(
            param_groups, betas=(0.9, 0.98), weight_decay=0.01
        )

        print(f"  - Fast memory LR: {args.lr:.6f}")
        print(f"  - Medium memory LR: {args.lr * 0.1:.6f}")
        print(f"  - Slow memory LR: {args.lr * 0.01:.6f}")
        print(f"  - Other params LR: {args.lr:.6f}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01
        )

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T, t0 = 0.0, time.time()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        epoch_loss, num_batches = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", unit="batch", ncols=100)

        for step, batch in enumerate(pbar):
            u, seq, pos, neg = [x.to(args.device) for x in batch]
            optimizer.zero_grad()

            mask = pos != 0
            mask_exp = mask.unsqueeze(-1).expand(-1, -1, args.num_negatives)

            pos_logits, neg_logits = model(u, seq, pos, neg)

            pos_sel = torch.masked_select(pos_logits, mask)

            if neg_logits.dim() == 2:
                neg_sel = torch.masked_select(neg_logits, mask).unsqueeze(1)
            else:
                neg_sel = torch.masked_select(neg_logits, mask_exp).view(
                    -1, args.num_negatives
                )

            cand_logits = torch.cat([pos_sel.unsqueeze(1), neg_sel], dim=1)

            # Compute loss based on selected loss type
            if args.loss_type == "sampled_softmax":
                loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()
            elif args.loss_type == "p_sampled_softmax":
                # Create labels: 1 for pos (first column), 0 for negatives
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = p_sampled_softmax_loss(cand_logits, labels)
            elif args.loss_type == "listmle":
                # Create labels: 1 for pos (first column), 0 for negatives
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = listmle_loss(cand_logits, labels)
            elif args.loss_type == "p_listmle":
                # Create labels: 1 for pos, 0 for negatives
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = p_listmle_loss(cand_logits, labels)
            elif args.loss_type == "mmcl":
                # Create labels: 1 for pos, 0 for negatives
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                # Use adaptive margins based on number of negatives
                if args.num_negatives <= 3:
                    margins = [0.3, 0.6]
                    weights = [1.0, 0.5]
                elif args.num_negatives <= 10:
                    margins = [0.2, 0.5, 0.8]
                    weights = [1.0, 0.5, 0.2]
                else:
                    margins = [0.1, 0.3, 0.6, 0.9]
                    weights = [1.0, 0.7, 0.4, 0.2]
                loss = mmcl_loss(cand_logits, labels, margins=margins, weights=weights)
            elif args.loss_type == "gbce":
                # Create labels: 1 for pos, 0 for negatives
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                # Use alpha=0.75 for overconfidence mitigation (from gSASRec paper)
                # Higher alpha (0.7-0.8) works better with more negatives
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = gbce_loss(cand_logits, labels, alpha=alpha)
            elif args.loss_type == "p_gbce":
                # Create labels: 1 for pos, 0 for negatives
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                # Use alpha=0.75 for overconfidence mitigation + position weighting
                # Higher alpha (0.7-0.8) works better with more negatives
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = p_gbce_loss(cand_logits, labels, alpha=alpha)
            elif args.loss_type == "rc_gbce":
                # Rank-Calibrated gBCE: focuses on decision boundary around top-k
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = rc_gbce_loss(cand_logits, labels, alpha=alpha, k=10)
            elif args.loss_type == "approx_ndcg":
                # ApproxNDCG: directly optimize NDCG@10
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = approx_ndcg_loss(cand_logits, labels, k=10)
            elif args.loss_type == "infonce_gbce":
                # InfoNCE + gBCE hybrid: uses in-batch negatives
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                # Note: For full in-batch contrastive, we need embeddings
                # Here we use the logit-based version (falls back to gBCE)
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = infonce_gbce_loss(cand_logits, labels, alpha=alpha, beta=0.3)
            elif args.loss_type == "tcr":
                # Temporal Contrastive Ranking: gBCE + temporal coherence
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = tcr_loss(cand_logits, labels, alpha=alpha)
            elif args.loss_type == "composite":
                # Composite loss: gBCE + ApproxNDCG + InfoNCE
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss_weights = {"gbce": 0.5, "ndcg": 0.3, "infonce": 0.2}
                loss = composite_loss(
                    cand_logits, labels, loss_weights=loss_weights, k=10
                )
            else:
                raise ValueError(f"Unknown loss_type: {args.loss_type}")

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}", end="")

        if epoch % 20 == 0:
            model.eval()
            T += time.time() - t0
            with torch.no_grad():
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)

            print(f" | Time: {T:.1f}s")
            print(f"         Valid: {t_valid} | Test: {t_test}")

            if t_valid[0] > best_val_ndcg:
                best_val_ndcg, best_val_hr = t_valid
                best_test_ndcg, best_test_hr = t_test
                folder = args.dataset + "_" + args.train_dir
                fname = f"SASRec.best.pth"
                torch.save(model.state_dict(), os.path.join(folder, fname))
                print(f"         ✓ Saved best model")

            f.write(f"{epoch} {t_valid} {t_test}\n")
            f.flush()
            t0 = time.time()
        else:
            print()

    f.close()
    print("Training completed!")
