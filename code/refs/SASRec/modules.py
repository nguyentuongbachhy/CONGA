import torch
import torch.nn as nn


class PointwiseFeedForward(nn.Module):
    """Point-wise feed-forward network (no residual; handled by SASRecBlock)."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class SASRecBlock(nn.Module):
    """One SASRec transformer block with pre-norm, causal self-attention and FFN."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-8)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-8)
        self.ffn = PointwiseFeedForward(d_model, dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm causal self-attention
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.attn_dropout(attn_out)

        # Pre-norm FFN
        residual = x
        x = residual + self.ffn_dropout(self.ffn(self.ffn_norm(x)))

        return x

