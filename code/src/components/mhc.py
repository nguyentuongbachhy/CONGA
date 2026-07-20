import torch
import torch.nn as nn
from typing import Tuple, Any, Optional, cast
import mhc_cuda


class MHCFusedFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(
        ctx: Any,
        x_streams: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_bias: torch.Tensor,
        rms_weight: torch.Tensor,
        alpha_pre: torch.Tensor,
        alpha_post: torch.Tensor,
        alpha_res: torch.Tensor,
        rms_eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H_pre, H_post, H_res, proj_raw, rstd = mhc_cuda.fused_forward(
            x_streams, proj_weight, proj_bias, rms_weight,
            alpha_pre, alpha_post, alpha_res, rms_eps
        )
        ctx.save_for_backward(
            x_streams, proj_weight, proj_bias, rms_weight,
            H_pre, H_post, proj_raw, rstd,
            alpha_pre, alpha_post, alpha_res
        )
        ctx.rms_eps = rms_eps
        return H_pre, H_post, H_res

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx: Any, grad_H_pre: torch.Tensor, grad_H_post: torch.Tensor, grad_H_res: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        (x_streams, proj_weight, proj_bias, rms_weight,
         H_pre, H_post, proj_raw, rstd,
         alpha_pre, alpha_post, alpha_res) = ctx.saved_tensors

        grads = mhc_cuda.fused_backward(
            grad_H_pre, grad_H_post, grad_H_res,
            x_streams, proj_weight, proj_bias, rms_weight,
            H_pre, H_post, proj_raw, rstd,
            alpha_pre, alpha_post, alpha_res, ctx.rms_eps
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
            None
        )


class MHCLayer(nn.Module):
    def __init__(self, hidden_units: int, num_streams: int = 4, init_scale: float = 0.01):
        super().__init__()
        assert num_streams == 4, "MHC currently only supports 4 streams due to kernel constraints"
        self.C = hidden_units
        self.n = num_streams
        self.input_dim = self.n * self.C

        self.alpha_pre = nn.Parameter(torch.tensor(init_scale))
        self.alpha_post = nn.Parameter(torch.tensor(init_scale))
        self.alpha_res = nn.Parameter(torch.tensor(init_scale))

        total_out = self.n + self.n + 2  # 4 pre + 4 post + 2 Kronecker factors
        self.fused_proj = nn.Linear(self.input_dim, total_out)
        nn.init.normal_(self.fused_proj.weight, std=0.02)
        nn.init.zeros_(self.fused_proj.bias)
        self.rms_weight = nn.Parameter(torch.ones(self.input_dim))
        self.rms_eps = 1e-6

    def forward(self, x_streams: torch.Tensor, layer_fn: nn.Module) -> torch.Tensor:
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
                self.rms_eps
            )
        )
        H_pre, H_post, H_res = result

        res_branch = torch.matmul(H_res, x_streams)
        agg_input = torch.matmul(H_pre.unsqueeze(2), x_streams).squeeze(2)
        func_output = layer_fn(agg_input)
        post_branch = H_post.unsqueeze(-1) * func_output.unsqueeze(2)

        return res_branch + post_branch
