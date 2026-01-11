import torch
from typing import Any, Tuple

import swiglu_cuda


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, w1_out: torch.Tensor, w2_out: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(w1_out, w2_out)
        return swiglu_cuda.fwd(w1_out, w2_out)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w1_out, w2_out = ctx.saved_tensors
        grad_w1, grad_w2 = swiglu_cuda.bwd(grad_out.contiguous(), w1_out, w2_out)
        return grad_w1, grad_w2


class SwiGLU(torch.nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_units, hidden_units * 4, bias=False)
        self.w2 = torch.nn.Linear(hidden_units, hidden_units * 4, bias=False)
        self.w3 = torch.nn.Linear(hidden_units * 4, hidden_units, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = self.w1(x)
        w2_out = self.w2(x)
        hidden = SwiGLUFunction.apply(w1_out, w2_out)
        return self.w3(self.dropout(hidden))