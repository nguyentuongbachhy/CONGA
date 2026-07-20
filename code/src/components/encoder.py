from typing import Optional, Callable
import torch
import torch.nn.functional as F

from .rope import apply_rotary_pos_emb

class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.hidden_units: int = hidden_units
        self.num_heads: int = num_heads
        
        self.head_dim: int = hidden_units // num_heads

        assert self.head_dim * num_heads == hidden_units, "Hidden units must be divisible by num_heads"
        
        self.qkv_proj: torch.nn.Linear = torch.nn.Linear(hidden_units, hidden_units * 3, bias=False)
        
        self.out_proj: torch.nn.Linear = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        rotary_emb_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        B, L, H = x.shape

        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if rotary_emb_fn is not None:
            cos, sin = rotary_emb_fn(L, x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build the attention mask. Priority:
        #   1. explicit `attn_mask` (already broadcastable additive / bool)
        #   2. `key_padding_mask` (B, L) combined with a causal triangular mask
        #   3. neither -> fall back to sdpa's fused causal path
        if attn_mask is not None:
            final_mask = attn_mask
            use_causal = False
        elif key_padding_mask is not None:
            # BSARec-style additive mask with a large negative sentinel
            # (NOT -inf, to keep softmax numerically stable when an entire
            # row ends up masked — e.g. a fully-padded query position).
            neg = torch.tensor(-1e4, dtype=q.dtype, device=q.device)
            causal_row = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=q.device), diagonal=1
            )
            # (1, 1, L, L) causal additive
            causal_add = torch.zeros(1, 1, L, L, dtype=q.dtype, device=q.device)
            causal_add.masked_fill_(causal_row[None, None], neg)
            # (B, 1, 1, L) key padding additive, broadcast to (B, 1, L, L)
            pad_add = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
            pad_add.masked_fill_(key_padding_mask[:, None, None, :], neg)
            final_mask = causal_add + pad_add
            use_causal = False
        else:
            final_mask = None
            use_causal = True

        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=final_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=use_causal,
        )

        context = context.transpose(1, 2).contiguous().view(B, L, H)

        output = self.out_proj(context)

        # Zero out padding query positions to prevent leakage.
        # With a finite sentinel (-1e4) instead of -inf in the attn mask,
        # padding queries (q=0) produce softmax(uniform) = 1/L, which attends
        # to valid V vectors → non-zero output → exploding gradients in MHC LN.
        if key_padding_mask is not None:
            output = output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return output