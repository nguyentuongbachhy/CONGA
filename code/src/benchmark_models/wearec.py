import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmark_models._abstract_model import SequentialRecModel
from benchmark_models._modules import LayerNorm, FeedForward

"""
[Paper]
Author: Huayang Xu et al.
Title: "Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation"
Conference: AAAI 2026

[Code Reference]
https://github.com/hyxu2006/WEARec
(Ported from WEARec's standalone interface to the local SequentialRecModel interface.)
"""


class WEARecLayer(nn.Module):
    """Dynamic frequency-domain filtering + Haar wavelet enhancement.

    Uses `args.alpha` as the fixed blending weight (wavelet vs FFT),
    matching the published best configs.  `args.num_attention_heads` controls
    the number of multi-head projection heads.
    """

    def __init__(self, args):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // self.num_heads
        if args.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.seq_len = args.max_seq_length
        self.freq_bins = args.max_seq_length // 2 + 1
        self.alpha = args.alpha  # wavelet-vs-FFT blend (paper: 0.1~0.5)

        # Haar wavelet detail modulation weights: (1, num_heads, seq_len//2, head_dim)
        self.complex_weight = nn.Parameter(
            torch.randn(1, self.num_heads, args.max_seq_length // 2,
                        self.head_dim, dtype=torch.float32) * 0.02
        )

        # Frequency-domain base filter and bias (one per head + freq bin).
        self.base_filter = nn.Parameter(torch.ones(self.num_heads, self.freq_bins, 1))
        self.base_bias = nn.Parameter(
            torch.full((self.num_heads, self.freq_bins, 1), -0.1)
        )

        # Adaptive MLP: produces 2 modulation values per (head, freq_bin).
        self.adaptive_mlp = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.GELU(),
            nn.Linear(args.hidden_size, self.num_heads * self.freq_bins * 2),
        )

        # Gate blend (always used; alpha is fixed, not a sigmoid gate).
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    # ------------------------------------------------------------------
    def _wavelet(self, x_heads: torch.Tensor) -> torch.Tensor:
        """Single-level Haar wavelet decomposition + reconstruction."""
        B, H, N, D = x_heads.shape
        N_even = N if N % 2 == 0 else N - 1
        x = x_heads[:, :, :N_even, :]

        approx = 0.5 * (x[:, :, 0::2, :] + x[:, :, 1::2, :])
        detail = 0.5 * (x[:, :, 0::2, :] - x[:, :, 1::2, :])
        detail = detail * self.complex_weight  # learnable detail scaling

        out = torch.zeros_like(x_heads)
        out[:, :, 0::2, :] = approx + detail
        out[:, :, 1::2, :] = approx - detail
        if N_even < N:
            out[:, :, -1:, :] = x_heads[:, :, -1:, :]
        return out

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        B, L, D = input_tensor.shape
        # Reshape to (B, num_heads, L, head_dim)
        x_h = input_tensor.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ---- (1) Global: FFT + adaptive multi-learnable filter ----
        F_fft = torch.fft.rfft(x_h, dim=2, norm="ortho")  # (B, H, freq_bins, head_dim)

        context = input_tensor.mean(dim=1)  # (B, D)
        adapt = self.adaptive_mlp(context).view(B, self.num_heads, self.freq_bins, 2)
        eff_filter = self.base_filter * (1 + adapt[..., 0:1])
        eff_bias = self.base_bias + adapt[..., 1:2]
        F_mod = F_fft * eff_filter + eff_bias
        x_fft = torch.fft.irfft(F_mod, dim=2, n=self.seq_len, norm="ortho")

        # ---- (2) Local: Haar wavelet ----
        x_wav = self._wavelet(x_h)

        # ---- (3) Blend ----
        x_comb = (1.0 - self.alpha) * x_wav + self.alpha * x_fft

        # Merge heads back: (B, H, L, head_dim) -> (B, L, D)
        x_out = x_comb.permute(0, 2, 1, 3).reshape(B, L, D)
        hidden = self.out_dropout(x_out) + input_tensor
        return self.LayerNorm(hidden)


class WEARecBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = WEARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.layer(x))


class WEARecEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        block = WEARecBlock(args)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class WEARecModel(SequentialRecModel):
    """WEARec (AAAI 2026) adapted to the local benchmark interface.

    Relevant CLI args (existing):
        --num_heads   number of multi-head projection heads (paper: 2 or 8)
        --alpha       wavelet-vs-FFT blend ratio            (paper: 0.1~0.5)
        --dropout_rate
        --hidden_units / --num_blocks / --maxlen
    """

    def __init__(self, args):
        super().__init__(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = WEARecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids: torch.Tensor, user_ids=None,
                all_sequence_output: bool = False) -> torch.Tensor:
        x = self.add_position_embedding(input_ids)          # (B, L, H)
        return self.item_encoder(x)                         # (B, L, H)

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_out = self.forward(input_ids)  # (B, L, H)
        gamma = 1e-10

        if answers.dim() == 2:
            # Seq-level: supervise all non-padding positions with BPR.
            pos_emb = self.item_embeddings(answers)
            neg_emb = self.item_embeddings(neg_answers)
            pos_logits = (pos_emb * seq_out).sum(-1)
            neg_logits = (neg_emb * seq_out).sum(-1)
            mask = (answers != 0)
            loss = -torch.log(gamma + torch.sigmoid(pos_logits - neg_logits))
            return loss[mask].mean()

        # Prefix path: last-position CE over full item set.
        seq_last = seq_out[:, -1, :]
        logits = torch.matmul(seq_last, self.item_embeddings.weight.T)
        return nn.CrossEntropyLoss()(logits, answers)
