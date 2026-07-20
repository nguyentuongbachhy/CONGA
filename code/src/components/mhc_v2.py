import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mhcv2_cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


class MHCv2FusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_streams, sublayer_out, post_scale, ln_weight, ln_bias, kron_logits):
        kron_probs = torch.sigmoid(kron_logits)
        # Save original sublayer dtype for backward grad casting
        ctx.sublayer_dtype = sublayer_out.dtype

        out, save_mean, save_rstd = mhcv2_cuda.fused_forward(
            x_streams if x_streams.is_contiguous() else x_streams.contiguous(),
            sublayer_out if sublayer_out.is_contiguous() else sublayer_out.contiguous(),
            post_scale if post_scale.is_contiguous() else post_scale.contiguous(),
            ln_weight if ln_weight.is_contiguous() else ln_weight.contiguous(),
            ln_bias if ln_bias.is_contiguous() else ln_bias.contiguous(),
            kron_probs if kron_probs.is_contiguous() else kron_probs.contiguous(),
        )
        ctx.save_for_backward(
            x_streams, sublayer_out, post_scale, ln_weight, ln_bias,
            kron_probs, save_mean, save_rstd,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x_streams, sublayer_out, post_scale, ln_weight, ln_bias,
         kron_probs, save_mean, save_rstd) = ctx.saved_tensors

        grads = mhcv2_cuda.fused_backward(
            grad_out if grad_out.is_contiguous() else grad_out.contiguous(),
            x_streams if x_streams.is_contiguous() else x_streams.contiguous(),
            sublayer_out if sublayer_out.is_contiguous() else sublayer_out.contiguous(),
            post_scale if post_scale.is_contiguous() else post_scale.contiguous(),
            ln_weight if ln_weight.is_contiguous() else ln_weight.contiguous(),
            ln_bias if ln_bias.is_contiguous() else ln_bias.contiguous(),
            kron_probs if kron_probs.is_contiguous() else kron_probs.contiguous(),
            save_mean, save_rstd,
        )
        grad_x, grad_sublayer, grad_ps, grad_lw, grad_lb, grad_kp = grads

        # Cast grad_sublayer back to original sublayer dtype
        grad_sublayer = grad_sublayer.to(ctx.sublayer_dtype)

        grad_kron_logits = grad_kp.float() * kron_probs.float() * (1.0 - kron_probs.float())

        return grad_x, grad_sublayer, grad_ps, grad_lw, grad_lb, grad_kron_logits


class MHCv2Layer(nn.Module):
    def __init__(self, hidden_units: int, num_streams: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.C = hidden_units
        self.N = num_streams
        self.K = int(math.log2(num_streams))
        assert 2 ** self.K == num_streams, "num_streams phải là lũy thừa của 2."

        self.pre_weights = nn.Parameter(torch.zeros(num_streams))

        self.post_scale = nn.Parameter(torch.ones(num_streams))

        self.ln_weight = nn.Parameter(torch.ones(num_streams, hidden_units))
        self.ln_bias = nn.Parameter(torch.zeros(num_streams, hidden_units))
        self.ln_eps = 1e-12

        self.kron_logits = nn.Parameter(torch.full((self.K,), 2.0))
        
        self.dropout = nn.Dropout(dropout)

    def _build_mix_matrix(self) -> torch.Tensor:
        probs = torch.sigmoid(self.kron_logits)
        p = probs[0]
        M = torch.stack([p, 1 - p, 1 - p, p]).reshape(2, 2)
        for i in range(1, self.K):
            p = probs[i]
            Mi = torch.stack([p, 1 - p, 1 - p, p]).reshape(2, 2)
            M = torch.kron(Mi, M)
        return M  # (N, N)

    def forward(self, x_streams: torch.Tensor, layer_fn) -> torch.Tensor:
        """
        x_streams: (B, L, N, C)
        layer_fn: Sublayer (Attention hoặc FFN) nhận (B, L, C)
        """
        # BƯỚC 1: Aggregation — softmax blend N streams -> 1
        attn_weights = F.softmax(self.pre_weights, dim=0)  # (N,)
        # matmul faster than einsum: (B,L,N,C) @ (N,) -> (B,L,C)
        agg_input = (x_streams * attn_weights.view(1, 1, self.N, 1)).sum(dim=2)

        sublayer_out = self.dropout(layer_fn(agg_input))  # (B, L, C)

        # BƯỚC 3-5: Fused Residual + LN + Kronecker Mix
        if _HAS_CUDA and x_streams.is_cuda:
            return MHCv2FusedFunction.apply(
                x_streams, sublayer_out,
                self.post_scale, self.ln_weight, self.ln_bias,
                self.kron_logits,
            )

        res_streams = x_streams + sublayer_out.unsqueeze(2) * self.post_scale.view(1, 1, self.N, 1)

        mean = res_streams.mean(dim=-1, keepdim=True)
        x_norm = (res_streams - mean) * torch.rsqrt(
            res_streams.var(dim=-1, keepdim=True, correction=0) + self.ln_eps
        )
        x_streams = x_norm * self.ln_weight + self.ln_bias

        # BƯỚC 5: Kronecker Mixing
        M = self._build_mix_matrix()
        return torch.einsum('blnc,mn->blmc', x_streams, M)