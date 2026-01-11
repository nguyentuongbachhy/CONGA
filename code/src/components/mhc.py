import torch
import torch.nn as nn
from typing import Tuple, Any, Optional, cast

import mhc_cuda


class MHCFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x_streams: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_bias: torch.Tensor,
        rms_weight: torch.Tensor,
        alpha_pre: torch.Tensor,
        alpha_post: torch.Tensor,
        alpha_res: torch.Tensor,
        rms_eps: float,
        n_iters: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H_pre, H_post, H_res, x_norm, proj_output, rstd = mhc_cuda.fused_forward(
            x_streams.contiguous(),
            proj_weight.contiguous(),
            proj_bias.contiguous(),
            rms_weight.contiguous(),
            alpha_pre.item(),
            alpha_post.item(),
            alpha_res.item(),
            rms_eps,
            n_iters
        )
        ctx.save_for_backward(
            x_streams, x_norm, proj_weight, rms_weight, 
            H_pre, H_post, proj_output, rstd,
            alpha_pre, alpha_post, alpha_res
        )
        ctx.n_iters = n_iters
        ctx.rms_eps = rms_eps
        return H_pre, H_post, H_res

    @staticmethod
    def backward(ctx: Any, grad_H_pre: torch.Tensor, grad_H_post: torch.Tensor, grad_H_res: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        (x_streams, x_norm, proj_weight, rms_weight, 
         H_pre, H_post, proj_output, rstd,
         alpha_pre, alpha_post, alpha_res) = ctx.saved_tensors
        
        grads = mhc_cuda.fused_backward(
            grad_H_pre.contiguous(),
            grad_H_post.contiguous(),
            grad_H_res.contiguous(),
            x_norm.contiguous(),
            x_streams.contiguous(),
            proj_weight.contiguous(),
            rms_weight.contiguous(),
            H_pre.contiguous(),
            H_post.contiguous(),
            proj_output.contiguous(),
            rstd.contiguous(),
            alpha_pre.item(),
            alpha_post.item(),
            alpha_res.item(),
            ctx.rms_eps,
            ctx.n_iters
        )
        
        grad_x, grad_proj_weight, grad_proj_bias, grad_rms_weight, grad_alpha_pre, grad_alpha_post, grad_alpha_res = grads
        
        return (
            grad_x,
            grad_proj_weight,
            grad_proj_bias,
            grad_rms_weight,
            grad_alpha_pre.squeeze(),
            grad_alpha_post.squeeze(),
            grad_alpha_res.squeeze(),
            None,
            None
        )


class MHCLayer(nn.Module):
    def __init__(self, hidden_units: int, num_streams: int = 4, init_scale: float = 0.01, n_iters: int = 3):
        super().__init__()
        self.C = hidden_units
        self.n = num_streams
        self.n_iters = n_iters
        self.input_dim = self.n * self.C
        
        self.alpha_pre = nn.Parameter(torch.tensor(init_scale))
        self.alpha_post = nn.Parameter(torch.tensor(init_scale))
        self.alpha_res = nn.Parameter(torch.tensor(init_scale))
        
        total_out = self.n + self.n + self.n * self.n
        self.fused_proj = nn.Linear(self.input_dim, total_out)
        self.rms_weight = nn.Parameter(torch.ones(self.input_dim))
        self.rms_eps = 1e-6

    def forward(self, x_streams: torch.Tensor, layer_fn: nn.Module) -> torch.Tensor:
        B, L, n, C = x_streams.shape
        
        result = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            MHCFusedFunction.apply(
                x_streams,
                self.fused_proj.weight,
                self.fused_proj.bias,
                self.rms_weight,
                self.alpha_pre,
                self.alpha_post,
                self.alpha_res,
                self.rms_eps,
                self.n_iters
            )
        )
        H_pre, H_post, H_res = result

        res_branch = torch.matmul(H_res, x_streams)
        agg_input = torch.matmul(H_pre.unsqueeze(2), x_streams).squeeze(2)
        func_output = layer_fn(agg_input)
        post_branch = H_post.unsqueeze(-1) * func_output.unsqueeze(2)

        return res_branch + post_branch