"""
Rotary Position Embedding (RoPE) implementation.

Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.
    
    Applies rotary position embeddings to query and key tensors in attention.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self.cached_cos: torch.Tensor | None = None
        self.cached_sin: torch.Tensor | None = None
        self.cached_seq_len: int = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute or retrieve cached cos/sin values."""
        if self.cached_cos is None or seq_len > self.cached_seq_len:
            self.cached_seq_len = max(seq_len, self.cached_seq_len)
            
            t = torch.arange(self.cached_seq_len, dtype=torch.float32, device=device)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self.cached_cos = emb.cos()[None, :, None, :]
            self.cached_sin = emb.sin()[None, :, None, :]
        
        return self.cached_cos[:, :seq_len, ...], self.cached_sin[:, :seq_len, ...]
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary embeddings for given sequence length.
        
        Args:
            seq_len: Sequence length
            device: Device to place tensors on
            
        Returns:
            cos, sin: Cosine and sine embeddings [1, seq_len, 1, dim]
        """
        return self._compute_cos_sin(seq_len, device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    Args:
        x: Input tensor [..., dim]
        
    Returns:
        Rotated tensor [..., dim]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]
        cos: Cosine embeddings [1, seq_len, 1, head_dim]
        sin: Sine embeddings [1, seq_len, 1, head_dim]
        
    Returns:
        q_embed, k_embed: Rotated query and key tensors
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
