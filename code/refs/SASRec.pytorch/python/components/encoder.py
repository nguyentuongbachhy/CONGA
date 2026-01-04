import math
from typing import Optional, Tuple, Callable
import torch

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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        
        context = context.transpose(1, 2).contiguous().view(B, L, H)
        
        output = self.out_proj(context)
        return output 