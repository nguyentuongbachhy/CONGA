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
        
        self.W_q: torch.nn.Linear = torch.nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_k: torch.nn.Linear = torch.nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_v: torch.nn.Linear = torch.nn.Linear(hidden_units, hidden_units, bias=False)
        
        self.out_proj: torch.nn.Linear = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], rotary_emb_fn: Optional[Callable] = None) -> torch.Tensor:
        B, L, H = x.shape

        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim)

        if rotary_emb_fn is not None:
            cos, sin = rotary_emb_fn(L, x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        use_causal = True if attn_mask is None else False
        
        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=use_causal
        )
        
        context = context.transpose(1, 2).contiguous().view(B, L, H)
        
        output = self.out_proj(context)
        return output