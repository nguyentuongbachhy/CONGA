import torch
import torch.nn.functional as F
from typing import Optional


def listmle_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    _, indices = y_true.sort(descending=True, dim=-1)
    preds_sorted_by_true = torch.gather(y_pred, dim=1, index=indices)
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    return torch.mean(torch.sum(observation_loss, dim=1))


def p_listmle_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    _, indices = y_true.sort(descending=True, dim=-1)
    preds_sorted_by_true = torch.gather(y_pred, dim=1, index=indices)
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    slate_length = y_pred.shape[1]
    positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=y_pred.device)
    position_weights = (1.0 / torch.log2(positions + 1.0)).unsqueeze(0)
    weighted_loss = observation_loss * position_weights
    return torch.mean(torch.sum(weighted_loss, dim=1))


def p_sampled_softmax_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    softmax_probs = F.log_softmax(y_pred, dim=1)
    base_loss = -softmax_probs[:, 0]
    slate_length = y_pred.shape[1]
    positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=y_pred.device)
    position_weights = (1.0 / torch.log2(positions + 1.0)).unsqueeze(0)
    weighted_loss = base_loss * position_weights[:, 0]
    return torch.mean(weighted_loss)


def mmcl_loss(y_pred: torch.Tensor, y_true: torch.Tensor, margins=[0.2, 0.5, 0.8], weights=[1.0, 0.5, 0.2], temperature=1.0, eps=1e-10) -> torch.Tensor:
    y_pred_scaled = y_pred / temperature
    pos_sim = y_pred_scaled[:, 0]
    neg_sims = y_pred_scaled[:, 1:]
    pos_loss = torch.mean(torch.relu(1.0 - pos_sim))
    neg_loss = 0.0
    for margin, weight in zip(margins, weights):
        margin_term = neg_sims - margin
        neg_loss += weight * torch.mean(F.softplus(margin_term))
    return pos_loss + neg_loss


def gbce_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha=0.75, temperature=1.0, eps=1e-10) -> torch.Tensor:
    y_pred_scaled = y_pred / temperature
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_loss = -(y_true * torch.log(sigmoid_pred + eps) + (1 - y_true) * torch.log(1 - sigmoid_pred + eps))
    bce_loss = torch.mean(bce_loss)
    ce_loss = -F.log_softmax(y_pred_scaled, dim=1)[:, 0]
    ce_loss = torch.mean(ce_loss)
    return alpha * bce_loss + (1 - alpha) * ce_loss


def p_gbce_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha=0.75, temperature=1.0, eps=1e-10) -> torch.Tensor:
    y_pred_scaled = y_pred / temperature
    slate_length = y_pred.shape[1]
    positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=y_pred.device)
    position_weights = (1.0 / torch.log2(positions + 1.0)).unsqueeze(0)
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_loss_per_item = -(y_true * torch.log(sigmoid_pred + eps) + (1 - y_true) * torch.log(1 - sigmoid_pred + eps))
    weighted_bce_loss = torch.mean(bce_loss_per_item * position_weights)
    ce_loss_per_item = -F.log_softmax(y_pred_scaled, dim=1)
    weighted_ce_loss = ce_loss_per_item * position_weights
    positive_mask = y_true == 1.0
    weighted_ce_loss = torch.mean(weighted_ce_loss * positive_mask)
    return alpha * weighted_bce_loss + (1 - alpha) * weighted_ce_loss


def rc_gbce_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha=0.75, k=10, temperature=1.0, eps=1e-10) -> torch.Tensor:
    y_pred_scaled = y_pred / temperature
    pred_diff = y_pred_scaled.unsqueeze(-1) - y_pred_scaled.unsqueeze(-2)
    approx_ranks = 1.0 + torch.sigmoid(pred_diff * 10.0).sum(dim=-1)
    rank_weights = torch.exp(-((approx_ranks - k) ** 2) / (2.0 * k))
    rank_weights = rank_weights / (rank_weights.sum(dim=-1, keepdim=True) + eps)
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_per_item = -(y_true * torch.log(sigmoid_pred + eps) + (1 - y_true) * torch.log(1 - sigmoid_pred + eps))
    weighted_bce = (bce_per_item * rank_weights).sum(dim=-1).mean()
    ce_per_item = -F.log_softmax(y_pred_scaled, dim=1)
    positive_mask = y_true == 1.0
    weighted_ce = (ce_per_item * rank_weights * positive_mask).sum(dim=-1).mean()
    return alpha * weighted_bce + (1 - alpha) * weighted_ce


def approx_ndcg_loss(y_pred: torch.Tensor, y_true: torch.Tensor, temperature=1.0, k=10, eps=1e-10) -> torch.Tensor:
    y_pred_scaled = y_pred / temperature
    _, slate_length = y_pred.shape
    pred_diff = y_pred_scaled.unsqueeze(-1) - y_pred_scaled.unsqueeze(-2)
    soft_ranks = 1.0 + torch.sigmoid(pred_diff * 5.0).sum(dim=-1)
    soft_ranks = soft_ranks.clamp(min=1.0, max=float(slate_length))
    discounts = 1.0 / torch.log2(soft_ranks + 1.0)
    gains = y_true
    topk_weight = torch.sigmoid((k + 0.5 - soft_ranks) * 2.0)
    dcg = (gains * discounts * topk_weight).sum(dim=-1)
    sorted_gains, _ = gains.sort(dim=-1, descending=True)
    ideal_positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=y_pred.device)
    ideal_discounts = 1.0 / torch.log2(ideal_positions + 1.0)
    k_actual = min(k, slate_length)
    idcg = (sorted_gains[:, :k_actual] * ideal_discounts[:k_actual]).sum(dim=-1)
    ndcg = dcg / (idcg + eps)
    return -ndcg.mean()


def infonce_gbce_loss(y_pred: torch.Tensor, y_true: torch.Tensor, seq_emb: Optional[torch.Tensor] = None, pos_emb: Optional[torch.Tensor] = None, alpha=0.75, beta=0.3, temperature=0.1, gbce_temp=1.0, eps=1e-10) -> torch.Tensor:
    gbce = gbce_loss(y_pred, y_true, alpha=alpha, temperature=gbce_temp, eps=eps)
    if seq_emb is None or pos_emb is None:
        return gbce
    seq_emb_norm = F.normalize(seq_emb, p=2, dim=-1)
    pos_emb_norm = F.normalize(pos_emb, p=2, dim=-1)
    pos_sim = (seq_emb_norm * pos_emb_norm).sum(dim=-1) / temperature
    all_sim = torch.mm(seq_emb_norm, pos_emb_norm.t()) / temperature
    infonce = -pos_sim + torch.logsumexp(all_sim, dim=-1)
    infonce = infonce.mean()
    return gbce + beta * infonce


def tcr_loss(y_pred: torch.Tensor, y_true: torch.Tensor, seq_emb: Optional[torch.Tensor] = None, alpha=0.75, lambda_temporal=0.2, lambda_coherence=0.1, temperature=1.0, eps=1e-10) -> torch.Tensor:
    base_loss = gbce_loss(y_pred, y_true, alpha=alpha, temperature=temperature, eps=eps)
    if seq_emb is None or seq_emb.dim() != 3:
        return base_loss
    _, seq_len, _ = seq_emb.shape
    if seq_len < 3:
        return base_loss
    curr_emb = seq_emb[:, :-1, :]
    next_emb = seq_emb[:, 1:, :]
    curr_norm = F.normalize(curr_emb, p=2, dim=-1)
    next_norm = F.normalize(next_emb, p=2, dim=-1)
    pos_sim = (curr_norm * next_norm).sum(dim=-1)
    if seq_len > 3:
        distant_emb = seq_emb[:, 2:, :]
        distant_norm = F.normalize(distant_emb, p=2, dim=-1)
        neg_sim = (curr_norm[:, :-1, :] * distant_norm).sum(dim=-1)
        margin = 0.3
        temporal_loss = torch.relu(neg_sim - pos_sim[:, :-1] + margin).mean()
    else:
        temporal_loss = torch.tensor(0.0, device=y_pred.device)
    emb_diff = next_emb - curr_emb
    diff_mean = emb_diff.mean(dim=1, keepdim=True)
    coherence_loss = ((emb_diff - diff_mean) ** 2).mean()
    return base_loss + lambda_temporal * temporal_loss + lambda_coherence * coherence_loss


def duorec_cl_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 1.0,
    sim: str = "dot",
) -> torch.Tensor:
    """InfoNCE contrastive loss matching DuoRec's `info_nce` (WSDM'22)."""
    
    B = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    
    if sim == "cos":
        z = F.normalize(z, dim=-1)
    elif sim != "dot":
        raise ValueError(f"Unknown sim='{sim}', expected 'dot' or 'cos'.")

    # (2B, 2B) raw similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature

    # Mask out self-similarity by setting the main diagonal to -infinity.
    # This prevents the anchor from being used as its own negative.
    sim_matrix.fill_diagonal_(-float("inf"))

    # Labels represent the absolute index of the positive pair for each anchor.
    # z1 anchors (0 to B-1) match with z2 (B to 2B-1).
    # z2 anchors (B to 2B-1) match with z1 (0 to B-1).
    device = z.device
    labels = torch.cat([
        torch.arange(B, 2 * B, device=device),
        torch.arange(B, device=device)
    ])

    # F.cross_entropy efficiently handles the full matrix and absolute indices
    return F.cross_entropy(sim_matrix, labels)


def composite_loss(y_pred: torch.Tensor, y_true: torch.Tensor, seq_emb: Optional[torch.Tensor] = None, pos_emb: Optional[torch.Tensor] = None, loss_weights: dict = {}, k=10, eps=1e-10) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=y_pred.device)

    if loss_weights.get("gbce", 0) > 0:
        gbce = gbce_loss(y_pred, y_true, alpha=0.75, eps=eps)
        total_loss = total_loss + loss_weights["gbce"] * gbce

    if loss_weights.get("ndcg", 0) > 0:
        ndcg = approx_ndcg_loss(y_pred, y_true, k=k, eps=eps)
        total_loss = total_loss + loss_weights["ndcg"] * ndcg

    if loss_weights.get("infonce", 0) > 0 and seq_emb is not None and pos_emb is not None:
        seq_norm = F.normalize(seq_emb, p=2, dim=-1)
        pos_norm = F.normalize(pos_emb, p=2, dim=-1)

        tau = 0.1 

        logits_pos = (seq_norm * pos_norm).sum(dim=-1).unsqueeze(-1) / tau

        logits_all = torch.mm(seq_norm, pos_norm.t()) / tau

        infonce = (-logits_pos + torch.logsumexp(logits_all, dim=-1, keepdim=True)).mean()
        
        total_loss = total_loss + loss_weights["infonce"] * infonce
        
    return total_loss