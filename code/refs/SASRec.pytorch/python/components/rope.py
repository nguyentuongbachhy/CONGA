import torch
from typing import Tuple

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq: torch.Tensor = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        
        self.register_buffer("inv_freq", inv_freq)
        self.cached_cos: torch.Tensor | None = None
        self.cached_sin: torch.Tensor | None = None
        
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cached_cos is None or seq_len > self.cached_cos.shape[1]:
            t: torch.Tensor = torch.arange(seq_len, dtype=torch.float32, device=device)
            freqs: torch.Tensor = torch.einsum("i,j->ij", t, self.inv_freq)
            
            emb: torch.Tensor = torch.cat((freqs, freqs), dim=-1)
            
            self.cached_cos = emb.cos()[None, :, None, :]
            self.cached_sin = emb.sin()[None, :, None, :]
        
        assert self.cached_cos is not None and self.cached_sin is not None
        return self.cached_cos[:, :seq_len, ...], self.cached_sin[:, :seq_len, ...]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed