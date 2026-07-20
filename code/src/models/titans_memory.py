import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import titans_cuda

class TitansRecurrenceFunction(torch.autograd.Function):
    
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, k, v, q, alpha, theta, eta, M_init, S_init, use_cuda=True):
        ctx.use_cuda = use_cuda
        ctx.save_for_backward(k, v, q, alpha, theta, eta, M_init, S_init)
        if use_cuda:
            y = titans_cuda.recurrence_forward(k, v, q, alpha, theta, eta, M_init)
        else:
            y = _recurrence_pytorch_forward(k, v, q, alpha, theta, eta, M_init, S_init)
        return y
    
    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dY):
        k, v, q, alpha, theta, eta, M_init, S_init = ctx.saved_tensors
        if ctx.use_cuda:
            grads = titans_cuda.recurrence_backward(
                k, v, q, alpha, theta, eta, M_init, S_init, dY
            )
            dK, dV, dQ, dalpha, dtheta, deta, dM_init, dS_init = grads
        else:
            raise NotImplementedError("Use autograd path for PyTorch fallback")
        return dK, dV, dQ, dalpha, dtheta, deta, dM_init, dS_init, None


def _recurrence_pytorch_forward(k, v, q, alpha, theta, eta, M_init, S_init):
    """Pure PyTorch implementation for debugging/fallback."""
    B, L, D = k.shape
    M = M_init.clone()
    S = S_init.clone()
    outputs = []

    for t in range(L):
        k_t = k[:, t]  # [B, D]
        v_t = v[:, t]
        q_t = q[:, t]
        alpha_t = alpha[:, t]
        theta_t = theta[:, t]
        eta_t = eta[:, t]

        # Compute error: e = M @ k - v
        error = torch.bmm(M, k_t.unsqueeze(2)).squeeze(2) - v_t  # [B, D]
        # Gradient of associative loss: ∇ℓ = e ⊗ k^T
        grad = torch.bmm(error.unsqueeze(2), k_t.unsqueeze(1))  # [B, D, D]

        # Update momentum: S = diag(η) @ S - diag(θ) @ ∇ℓ
        S = eta_t.unsqueeze(2) * S - theta_t.unsqueeze(2) * grad
        # Update memory: M = diag(1-α) @ M + S
        M = (1 - alpha_t).unsqueeze(2) * M + S

        # Retrieve: y = M @ q
        y_t = torch.bmm(M, q_t.unsqueeze(2)).squeeze(2)
        outputs.append(y_t)

    return torch.stack(outputs, dim=1)


def titans_recurrence(k, v, q, alpha, theta, eta, M_init, S_init=None, use_cuda=True):
    """Wrapper for TITANS recurrence with autograd support.
    
    Args:
        k, v, q: [B, L, D] - keys, values, queries
        alpha, theta, eta: [B, L, D] - gates
        M_init: [B, D, D] - initial memory
        S_init: [B, D, D] or None - initial momentum (defaults to zeros)
        use_cuda: bool - use CUDA kernel if available
        
    Returns:
        y: [B, L, D] - retrieved outputs
    """
    B, L, D = k.shape
    
    assert k.shape == v.shape == q.shape == (B, L, D), "K, V, Q shape mismatch"
    assert alpha.shape == theta.shape == eta.shape == (B, L, D), "Gates shape mismatch"
    assert M_init.shape == (B, D, D), f"M_init shape {M_init.shape} != ({B}, {D}, {D})"
    
    if S_init is None:
        S_init = torch.zeros(B, D, D, dtype=k.dtype, device=k.device)
    assert S_init.shape == (B, D, D), f"S_init shape {S_init.shape} != ({B}, {D}, {D})"
    
    if use_cuda:
        return TitansRecurrenceFunction.apply(k, v, q, alpha, theta, eta, M_init, S_init, use_cuda)
    return _recurrence_pytorch_forward(k, v, q, alpha, theta, eta, M_init, S_init)


class TitansMemory(nn.Module):
    """Neural Long-term Memory (arXiv:2501.00663 Section 3.1)

    Linear memory M ∈ R^{d_k × d_v}, updated via gradient descent
    with momentum and weight decay on associative loss:
        ℓ(M; x_t) = ||M @ k_t - v_t||²

    Update rules (Eq. 13-14):
        S_t = diag(η_t) @ S_{t-1} - diag(θ_t) @ ∇ℓ(M; x_t)
        M_t = diag(1-α_t) @ M_{t-1} + S_t

    Retrieve (Eq. 15): y_t = M_t @ q_t
    """

    def __init__(
        self,
        d_model: int,
        d_mem: Optional[int] = None,
        conv_kernel: int = 4,
        use_cuda: bool = True,
        gate_eps: float = 1e-6,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            d_model: input/output dimension
            d_mem: memory dimension (defaults to d_model, must be 32/64/128 for CUDA)
            conv_kernel: kernel size for temporal convolutions
            use_cuda: whether to use CUDA kernel (auto-detects availability)
            gate_eps: epsilon for numerical stability in gates
        """
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem or d_model
        self.gate_eps = gate_eps
        self.use_cuda = use_cuda

        # K/V/Q Projections
        self.W_K = nn.Linear(d_model, self.d_mem, bias=False)
        self.W_V = nn.Linear(d_model, self.d_mem, bias=False)
        self.W_Q = nn.Linear(d_model, self.d_mem, bias=False)

        # Temporal convolutions (depthwise)
        self.conv_k = nn.Conv1d(
            self.d_mem, self.d_mem, conv_kernel, 
            padding=conv_kernel - 1, groups=self.d_mem
        )
        self.conv_v = nn.Conv1d(
            self.d_mem, self.d_mem, conv_kernel, 
            padding=conv_kernel - 1, groups=self.d_mem
        )
        self.conv_q = nn.Conv1d(
            self.d_mem, self.d_mem, conv_kernel, 
            padding=conv_kernel - 1, groups=self.d_mem
        )

        # Gate networks (α: forget, θ: learning rate, η: momentum)
        self.W_alpha = nn.Linear(d_model, self.d_mem)
        self.W_theta = nn.Linear(d_model, self.d_mem)
        self.W_eta = nn.Linear(d_model, self.d_mem)

        # Gate bias init for stable recurrence over long sequences
        nn.init.constant_(self.W_alpha.bias, -3.0)  # sigmoid(-3)≈0.05
        nn.init.constant_(self.W_theta.bias, -6.0)  # softplus(-6)≈0.0025 (conservative LR)
        nn.init.constant_(self.W_eta.bias, 1.0)      # sigmoid(1)≈0.73 (moderate momentum)

        # Output projection with fixed scaling
        self.out_norm = nn.LayerNorm(self.d_mem)
        self.out_proj = nn.Linear(self.d_mem, d_model, bias=False)
        # Fixed scale=0.08: slightly more aggressive than 0.06 but below crash threshold 0.10
        # Empirically: 0.06=stable/small, 0.10=crashes, 0.08=sweet spot
        self.register_buffer('out_scale', torch.tensor(0.08))

        # Dropout for regularization
        self.mem_dropout = nn.Dropout(dropout)

        # Learnable initial states
        self.M_init = nn.Parameter(torch.zeros(self.d_mem, self.d_mem))
        self.S_init = nn.Parameter(torch.zeros(self.d_mem, self.d_mem))
        nn.init.xavier_uniform_(self.M_init)

        nn.init.zeros_(self.out_proj.weight)

    def _project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to K/V/Q with temporal convolutions.
        
        Args:
            x: [B, L, d_model]
            
        Returns:
            k, v, q: [B, L, d_mem] - normalized keys/queries, activated values
        """
        B, L, _ = x.shape
        
        # Linear projections
        k = self.W_K(x).transpose(1, 2)  # [B, d_mem, L]
        v = self.W_V(x).transpose(1, 2)
        q = self.W_Q(x).transpose(1, 2)

        # Temporal convolutions (causal via padding)
        k = self.conv_k(k)[:, :, :L].transpose(1, 2)  # [B, L, d_mem]
        v = self.conv_v(v)[:, :, :L].transpose(1, 2)
        q = self.conv_q(q)[:, :, :L].transpose(1, 2)

        # Non-linearity
        k = F.silu(k)
        v = F.silu(v)
        q = F.silu(q)

        # Normalize keys and queries for stability
        k = F.normalize(k, dim=-1, eps=self.gate_eps)
        q = F.normalize(q, dim=-1, eps=self.gate_eps)

        return k, v, q

    def _compute_gates(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute α (forget), θ (lr), η (momentum) gates.
        
        Args:
            x: [B, L, d_model]
            
        Returns:
            alpha: [B, L, d_mem] ∈ (0, 1) - forget gate
            theta: [B, L, d_mem] ∈ (0, ∞) - learning rate
            eta: [B, L, d_mem] ∈ (0, 1) - momentum decay
        """
        # α: forget rate (higher = forget more of old memory)
        alpha = torch.sigmoid(self.W_alpha(x))
        # Clamp to avoid division by zero in backward (1-alpha should never be 0)
        alpha = torch.clamp(alpha, self.gate_eps, 1.0 - self.gate_eps)
        
        # θ: learning rate (positive, unbounded)
        theta = F.softplus(self.W_theta(x)) + self.gate_eps
        
        # η: momentum decay (similar to exponential moving average)
        eta = torch.sigmoid(self.W_eta(x))
        # Clamp to avoid division by zero in backward
        eta = torch.clamp(eta, self.gate_eps, 1.0 - self.gate_eps)
        
        return alpha, theta, eta

    def forward(
        self, 
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model] - input sequence
            padding_mask: [B, L] bool - True at **valid** (non-padding) positions.
                          When None every position is treated as valid.
            return_stats: if True, return dict with gate statistics
            
        Returns:
            output: [B, L, d_model] - gated output
            stats: dict (optional) - gate statistics for monitoring
        """
        B, L, D = x.shape
        
        # Project to K/V/Q
        k, v, q = self._project(x)
        
        # Compute gates
        alpha, theta, eta = self._compute_gates(x)

        # Zero-out gates at padding positions so the recurrence keeps
        # M and S unchanged: alpha=0 -> (1-α)=1 (no decay), theta=0 ->
        # no gradient update, eta=0 -> S contribution zeroed.
        if padding_mask is not None:
            valid = padding_mask.unsqueeze(-1).to(alpha.dtype)  # [B, L, 1]
            alpha = alpha * valid
            theta = theta * valid
            eta = eta * valid
        
        # Expand initial states to batch
        M_init = self.M_init.unsqueeze(0).expand(B, -1, -1).contiguous()
        S_init = self.S_init.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Run recurrence (CUDA or PyTorch)
        y = titans_recurrence(k, v, q, alpha, theta, eta, M_init, S_init, self.use_cuda)
        
        # Safety: prevent memory state explosion from causing NaN
        y = torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0)
        y = torch.clamp(y, -10.0, 10.0)
        
        # Output projection with fixed scaling
        y = self.out_norm(y)
        y = self.out_proj(y)
        y = self.mem_dropout(y)
        output = y * self.out_scale
        
        if return_stats:
            with torch.no_grad():
                stats = {
                    'alpha_mean': alpha.mean().item(),
                    'alpha_std': alpha.std().item(),
                    'theta_mean': theta.mean().item(),
                    'theta_std': theta.std().item(),
                    'eta_mean': eta.mean().item(),
                    'eta_std': eta.std().item(),
                    'output_scale': self.out_scale.item(),
                    'memory_init_norm': torch.norm(M_init).item(),
                    'using_cuda': self.use_cuda,
                }
            return output, stats
        
        return output


class TitansSASRec(nn.Module):
    """SASRec + TITANS neural memory via last-position fusion.

    Memory processes an EXTENDED sequence (e.g., 600 items) while attention
    processes a shorter window (e.g., 300 items). Memory output is added
    to the attention output at the LAST (prediction) position only.
    
    The memory uses a SINGLE SCALAR gate (not a full Linear layer) to 
    prevent the gate-explosion instability observed with learnable Linear gates.
    """

    def __init__(
        self, 
        base_model: nn.Module, 
        d_model: int, 
        d_mem: Optional[int] = None,
        use_cuda: bool = True,
        dropout: float = 0.2,
        maxlen: int = 300,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.maxlen = maxlen
        self.memory = TitansMemory(d_model, d_mem, use_cuda=use_cuda, dropout=dropout)
        
        # Position embeddings for memory (attention uses RoPE, memory needs its own)
        self.mem_pos_emb = nn.Embedding(max(maxlen, 1024) + 1, d_model, padding_idx=0)
        nn.init.normal_(self.mem_pos_emb.weight, std=0.02)
        self.mem_pos_emb.weight.data[0].zero_()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def _get_mem_feat(self, mem_seqs: torch.Tensor) -> torch.Tensor:
        item_ids = torch.as_tensor(mem_seqs, dtype=torch.long, device=self.base_model.dev)

        x = F.embedding(item_ids, self.base_model.item_emb.weight, padding_idx=0)
        
        valid_mask = (item_ids != 0)  # bool [B, L]
        positions = valid_mask.long().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        positions = positions * valid_mask.long()
        x = x + self.mem_pos_emb(positions)
        
        mem_out = self.memory(x, padding_mask=valid_mask)
        return mem_out[:, -1, :]

    def log2feats(self, log_seqs: Any, mem_seqs: Any = None, **kwargs) -> torch.Tensor:
        feats = self.base_model.log2feats(log_seqs, **kwargs)
        if mem_seqs is not None:
            mem_feat = self._get_mem_feat(mem_seqs)
            feats = feats.clone()
            feats[:, -1, :] = feats[:, -1, :] + mem_feat
        return feats

    def forward(
        self, 
        user_ids: Any, 
        log_seqs: Any, 
        pos_seqs: Any, 
        neg_seqs: Any,
        mem_seqs: Any = None,
    ) -> Tuple:
        log_feats = self.log2feats(log_seqs, mem_seqs=mem_seqs)

        pos_ids = torch.as_tensor(pos_seqs, dtype=torch.long, device=self.base_model.dev)
        pos_embs = F.embedding(pos_ids, self.base_model.item_emb.weight, padding_idx=0)

        neg_ids = torch.as_tensor(neg_seqs, dtype=torch.long, device=self.base_model.dev)
        neg_embs = F.embedding(neg_ids, self.base_model.item_emb.weight, padding_idx=0)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        if neg_embs.dim() == 3:
            neg_logits = (log_feats * neg_embs).sum(dim=-1)
        else:
            neg_logits = (log_feats.unsqueeze(-2) * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, log_feats, pos_embs[:, -1, :]

    def predict(self, user_ids: Any, log_seqs: Any, item_indices: Any, mem_seqs: Any = None) -> torch.Tensor:
        log_feats = self.log2feats(log_seqs, mem_seqs=mem_seqs)
        final_feat = log_feats[:, -1, :]

        item_ids = torch.as_tensor(item_indices, dtype=torch.long, device=self.base_model.dev)
        item_embs = F.embedding(item_ids, self.base_model.item_emb.weight, padding_idx=0)
        
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
