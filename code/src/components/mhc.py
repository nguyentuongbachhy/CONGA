import torch
import torch.nn as nn
from typing import Tuple, Any

import mhc_cuda  # type: ignore


class SinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, log_alpha: torch.Tensor, n_iters: int) -> torch.Tensor:
        x_input = log_alpha.float().contiguous()
        output = mhc_cuda.sinkhorn_forward(x_input, n_iters)
        ctx.save_for_backward(x_input)
        ctx.n_iters = n_iters
        return output.to(log_alpha.dtype)
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        log_alpha, = ctx.saved_tensors
        grad_input = mhc_cuda.sinkhorn_backward(
            grad_output.float().contiguous(),
            log_alpha.contiguous(),
            ctx.n_iters
        )
        return grad_input.to(grad_output.dtype), None


class SinkhornKnopp(nn.Module):
    def __init__(self, n_iters: int = 3):
        super().__init__()
        self.n_iters = n_iters
        
    def forward(self, log_alpha: torch.Tensor) -> torch.Tensor:
        shape = log_alpha.shape
        x_flat = log_alpha.view(-1, shape[-2], shape[-1])
        out_flat = SinkhornFunction.apply(x_flat, self.n_iters)
        return out_flat.view(shape)  # type: ignore[union-attr]
    
class MHCLayer(nn.Module):
    def __init__(self, hidden_units: int, num_streams: int = 4, init_scale: float = 0.01):
        super().__init__()
        self.C = hidden_units
        self.n = num_streams
        self.input_dim = self.n * self.C
        
        self.alpha_pre = nn.Parameter(torch.tensor(init_scale))
        self.alpha_post = nn.Parameter(torch.tensor(init_scale))
        self.alpha_res = nn.Parameter(torch.tensor(init_scale))
        
        self.out_features_split = [self.n, self.n, self.n * self.n]
        total_out = sum(self.out_features_split)
        self.fused_proj = nn.Linear(self.input_dim, total_out)
        
        if hasattr(nn, 'RMSNorm'):
            self.rms_norm = nn.RMSNorm(self.input_dim, eps=1e-6)
        else:
            self.rms_norm = nn.LayerNorm(self.input_dim, eps=1e-6)
        
        self.sinkhorn = SinkhornKnopp(n_iters=3)

    def forward(self, x_streams: torch.Tensor, layer_fn: nn.Module) -> torch.Tensor:
        B, L, n, C = x_streams.shape
        x_flat = x_streams.view(B, L, -1)
        
        with torch.amp.autocast_mode.autocast('cuda', enabled=False):
            x_norm = self.rms_norm(x_flat.float()).to(x_flat.dtype)

        projected = self.fused_proj(x_norm)

        proj_pre, proj_post, raw_res = torch.split(projected, self.out_features_split, dim=-1)

        raw_res = raw_res.view(B, L, n, n)

        H_pre = torch.sigmoid(self.alpha_pre * proj_pre)
        H_post = 2.0 * torch.sigmoid(self.alpha_post * proj_post)
        H_res = self.sinkhorn(self.alpha_res * raw_res) 

        res_branch = torch.matmul(H_res, x_streams)

        agg_input = torch.matmul(H_pre.unsqueeze(2), x_streams).squeeze(2)
        func_output = layer_fn(agg_input)
        post_branch = H_post.unsqueeze(-1) * func_output.unsqueeze(2)

        return res_branch + post_branch