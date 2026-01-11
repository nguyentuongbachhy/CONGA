import torch
from typing import Tuple, cast, Any

import rope_cuda


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
        
        self.register_buffer("cached_cos", emb.cos(), persistent=False)
        self.register_buffer("cached_sin", emb.sin(), persistent=False)
        
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cached_cos = cast(torch.Tensor, self.cached_cos)
        cached_sin = cast(torch.Tensor, self.cached_sin)
        
        if seq_len > cached_cos.shape[0]:
            t: torch.Tensor = torch.arange(seq_len, dtype=torch.float32, device=device)
            inv_freq = cast(torch.Tensor, self.inv_freq)
            freqs: torch.Tensor = torch.einsum("i,j->ij", t, inv_freq)
            emb: torch.Tensor = torch.cat((freqs, freqs), dim=-1)
            
            self.register_buffer("cached_cos", emb.cos(), persistent=False)
            self.register_buffer("cached_sin", emb.sin(), persistent=False)
            cached_cos = cast(torch.Tensor, self.cached_cos)
            cached_sin = cast(torch.Tensor, self.cached_sin)
        
        cos: torch.Tensor = cached_cos[:seq_len].to(device)
        sin: torch.Tensor = cached_sin[:seq_len].to(device)
        return cos, sin


class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(cos, sin)
        q_out, k_out = rope_cuda.fwd(q.contiguous(), k.contiguous(), cos.contiguous(), sin.contiguous())
        return q_out, k_out

    @staticmethod
    def backward(ctx: Any, grad_q: torch.Tensor, grad_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        dq, dk = rope_cuda.bwd(grad_q.contiguous(), grad_k.contiguous(), cos, sin)
        return dq, dk, None, None


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return cast(Tuple[torch.Tensor, torch.Tensor], RoPEFunction.apply(q, k, cos, sin))