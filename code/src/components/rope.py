import torch
from typing import Tuple, cast

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024) -> None:
        super().__init__()
        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        
        inv_freq: torch.Tensor = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t: torch.Tensor = torch.arange(max_seq_len, dtype=torch.float32)
        freqs: torch.Tensor = torch.einsum("i,j->ij", t, inv_freq)
        emb: torch.Tensor = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cached_cos", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("cached_sin", emb.sin()[None, :, None, :], persistent=False)
        
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cached_cos = cast(torch.Tensor, self.cached_cos)
        cached_sin = cast(torch.Tensor, self.cached_sin)
        
        if seq_len > cached_cos.shape[1]:
            t: torch.Tensor = torch.arange(seq_len, dtype=torch.float32, device=device)
            inv_freq = cast(torch.Tensor, self.inv_freq)
            freqs: torch.Tensor = torch.einsum("i,j->ij", t, inv_freq)
            emb: torch.Tensor = torch.cat((freqs, freqs), dim=-1)
            
            self.register_buffer("cached_cos", emb.cos()[None, :, None, :], persistent=False)
            self.register_buffer("cached_sin", emb.sin()[None, :, None, :], persistent=False)
            cached_cos = cast(torch.Tensor, self.cached_cos)
            cached_sin = cast(torch.Tensor, self.cached_sin)
        
        cos: torch.Tensor = cached_cos.to(device)
        sin: torch.Tensor = cached_sin.to(device)
        return cos[:, :seq_len, ...], sin[:, :seq_len, ...]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed