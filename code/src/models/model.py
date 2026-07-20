import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Optional

from components import SwiGLU, RotaryEmbedding, EncoderLayer
from components.mhc_v2 import MHCv2Layer


class PerBlockFreqLayer(nn.Module):
    """BSARec-faithful per-block frequency filter.

    Matches BSARec's FrequencyLayer exactly:
        output = LN(dropout(low_pass + beta² * high_pass) + x)

    Takes raw (unnormalised) block input x, applies the low-pass / high-pass
    decomposition, and returns a scale-stable output via internal residual + LN.
    This makes the output directly blendable with the attention path (which also
    has its own internal residual + LN in the BSARec-faithful _attn_fn path).
    """

    def __init__(self, hidden_units: int, cutoff: int, dropout_rate: float) -> None:
        super().__init__()
        self.cutoff = cutoff
        # randn init matches BSARec's FrequencyLayer initialisation
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_units))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: raw block input (B, L, H) — NOT pre-normalised."""
        B, L, C = x.shape
        x_fft = torch.fft.rfft(x.float(), dim=1, norm='ortho')
        cutoff = min(self.cutoff + 1, x_fft.size(1))
        x_fft_low = x_fft.clone()
        if cutoff < x_fft_low.size(1):
            x_fft_low[:, cutoff:, :] = 0
        x_low = torch.fft.irfft(x_fft_low, n=L, dim=1, norm='ortho').to(x.dtype)
        x_high = x - x_low
        seq_emb = self.dropout(x_low + self.sqrt_beta ** 2 * x_high)
        return self.layer_norm(seq_emb + x)  # BSARec: dropout(freq) + residual, then LN


class SASRec(torch.nn.Module):
    def __init__(self, user_num: int, item_num: int, args: Any) -> None:
        super(SASRec, self).__init__()

        self.user_num: int = user_num
        self.item_num: int = item_num
        self.dev: torch.device = args.device
        self.norm_first: bool = args.norm_first
        
        self.num_streams: int = getattr(args, 'num_streams', 4)
        self.num_blocks: int = args.num_blocks

        self.use_rope: bool = not getattr(args, 'no_rope', False)
        self.use_mhc: bool = not getattr(args, 'no_mhc', False)
        self.use_swiglu: bool = not getattr(args, 'no_swiglu', False)
        self.use_freq: bool = getattr(args, 'use_freq', False)
        self.fft_cutoff: int = getattr(args, 'fft_cutoff', 3)
        # 'post'     — single FrequencyLayer after all blocks (original v4)
        # 'parallel' — per-block FrequencyLayer parallel to attention (BSARec style)
        self.freq_mode: str = getattr(args, 'freq_mode', 'post')

        self.item_emb: torch.nn.Embedding = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)

        head_dim: int = args.hidden_units // args.num_heads
        self.rope: RotaryEmbedding = RotaryEmbedding(head_dim, max_seq_len=args.maxlen)

        if not self.use_rope:
            self.pos_emb: torch.nn.Embedding = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)

        # BSARec/DuoRec-style LayerNorm+Dropout applied to the raw token
        # embedding before the attention stack. Critical on sparse datasets
        # (Beauty/yelp) where padding dominates the sequence.
        self.input_layernorm: torch.nn.LayerNorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-12)
        self.emb_dropout: torch.nn.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms: torch.nn.ModuleList = torch.nn.ModuleList()
        self.attention_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.forward_layernorms: torch.nn.ModuleList = torch.nn.ModuleList()
        self.forward_layers: torch.nn.ModuleList = torch.nn.ModuleList()

        self.mhc_attn_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.mhc_ffn_layers: torch.nn.ModuleList = torch.nn.ModuleList()

        self.last_layernorm: torch.nn.LayerNorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # Frequency filter components.
        # post     mode: single post-encoder filter (v4 style).
        # parallel mode: per-block filter in parallel with attention (BSARec style).
        if self.use_freq:
            self.freq_alpha_val: float = getattr(args, 'freq_alpha', 0.3)
            if self.freq_mode == 'parallel':
                # One lightweight filter per transformer block.
                # alpha defaults to 0.7 to match BSARec's reported best.
                self.per_block_freq: torch.nn.ModuleList = torch.nn.ModuleList([
                    PerBlockFreqLayer(args.hidden_units, self.fft_cutoff, args.dropout_rate)
                    for _ in range(args.num_blocks)
                ])
            else:
                # post mode: original single-filter parameters
                self.freq_beta: torch.nn.Parameter = torch.nn.Parameter(
                    torch.ones(1, 1, args.hidden_units)
                )
                self.freq_out_dropout: torch.nn.Dropout = torch.nn.Dropout(p=args.dropout_rate)
                self.freq_layernorm: torch.nn.LayerNorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-12)

        # Learned stream fusion: replaces naive torch.mean across streams.
        # Each stream can carry different semantic signals; a learned linear
        # combination lets the model decide how to weight them.
        self.stream_fusion = torch.nn.Linear(self.num_streams, 1, bias=False)
        # Init to uniform 1/n so it starts equivalent to mean (smooth transition)
        torch.nn.init.constant_(self.stream_fusion.weight, 1.0 / self.num_streams)

        for i in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                EncoderLayer(args.hidden_units, args.num_heads, args.dropout_rate)
            )
            if self.use_mhc:
                self.mhc_attn_layers.append(
                    MHCv2Layer(args.hidden_units, num_streams=self.num_streams, dropout=args.dropout_rate)
                )

            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))

            if self.use_swiglu:
                self.forward_layers.append(
                    SwiGLU(args.hidden_units, args.dropout_rate)
                )
            else:
                self.forward_layers.append(nn.Sequential(
                    nn.Linear(args.hidden_units, args.hidden_units * 4),
                    nn.GELU(),
                    nn.Dropout(args.dropout_rate),
                    nn.Linear(args.hidden_units * 4, args.hidden_units),
                ))

            if self.use_mhc:
                self.mhc_ffn_layers.append(
                    MHCv2Layer(args.hidden_units, num_streams=self.num_streams, dropout=args.dropout_rate)
                )

    def _attn_fn(self, idx: int, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        rope_fn = self.rope if self.use_rope else None

        if self.use_freq and self.freq_mode == 'parallel' and not self.use_mhc:
            # BSARec-faithful path (only for no-MHC mode):
            # Both branches operate on raw x, each has its own internal
            # residual + LN (matching BSARec's architecture exactly).
            # PerBlockFreqLayer already returns LN(dropout(freq) + x).
            # Attention path mirrors BSARec's MultiHeadAttention: LN(attn(x) + x).
            # The combined output replaces x entirely; log2feats skips outer residual
            # for this mode (see the no_mhc loop below).
            attn_raw = self.attention_layers[idx](
                x, attn_mask=None, key_padding_mask=key_padding_mask, rotary_emb_fn=rope_fn
            )
            attn_out = self.attention_layernorms[idx](attn_raw + x)  # LN(attn + residual)
            freq_out = self.per_block_freq[idx](x)  # LN(dropout(freq) + residual)
            return self.freq_alpha_val * freq_out + (1.0 - self.freq_alpha_val) * attn_out

        if self.norm_first:
            x_in = self.attention_layernorms[idx](x)
        else:
            x_in = x
        attn_out = self.attention_layers[idx](
            x_in,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            rotary_emb_fn=rope_fn,
        )
        if not self.norm_first:
            attn_out = self.attention_layernorms[idx](attn_out)

        if self.use_freq and self.freq_mode == 'parallel':
            # MHC path: freq on pre-LN'd input; outer MHC residual handles skip.
            freq_out = self.per_block_freq[idx](x_in)
            return self.freq_alpha_val * freq_out + (1.0 - self.freq_alpha_val) * attn_out

        return attn_out
    
    def _ffn_fn(self, idx: int, x: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = self.forward_layernorms[idx](x)
        out = self.forward_layers[idx](x)
        if not self.norm_first:
            out = self.forward_layernorms[idx](out)
        return out

    def _apply_freq_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Post-encoder FrequencyLayer (freq_mode='post').

        Applied after all transformer blocks, before last_layernorm.
        """
        B, L, C = x.shape
        x_fft = torch.fft.rfft(x.float(), dim=1, norm='ortho')
        cutoff = min(self.fft_cutoff + 1, x_fft.size(1))
        x_fft_low = x_fft.clone()
        if cutoff < x_fft_low.size(1):
            x_fft_low[:, cutoff:, :] = 0
        x_low = torch.fft.irfft(x_fft_low, n=L, dim=1, norm='ortho').to(x.dtype)
        x_high = x - x_low
        dsp = self.freq_out_dropout(x_low + self.freq_beta ** 2 * x_high)
        freq_out = self.freq_layernorm(dsp + x)
        return (1.0 - self.freq_alpha_val) * x + self.freq_alpha_val * freq_out

    def log2feats(self, log_seqs: Any) -> torch.Tensor:
        item_ids: torch.Tensor = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)

        # key_padding_mask: True at padded positions (item_id == 0). Passed
        # into every attention layer so non-padding queries stop attending to
        # the zero-vector padding keys (which otherwise dilute softmax mass).
        key_padding_mask = (item_ids == 0)

        seqs = F.embedding(item_ids, self.item_emb.weight, padding_idx=0)

        if not self.use_rope:
            positions = torch.arange(1, seqs.size(1) + 1, device=self.dev).unsqueeze(0)
            seqs = seqs + self.pos_emb(positions)

        # Normalise the raw embedding (BSARec/DuoRec convention). Combined
        # with module-based init (std=0.02) this keeps activations bounded
        # at the first layer on sparse datasets.
        seqs = self.input_layernorm(seqs)
        seqs = self.emb_dropout(seqs)

        if self.use_mhc:
            x_streams = seqs.unsqueeze(2).repeat(1, 1, self.num_streams, 1)

            rope_fn = self.rope if self.use_rope else None
            for i in range(len(self.attention_layers)):
                # MHCv2 handles internal residual+LN; pass raw sublayers.
                x_streams = self.mhc_attn_layers[i](
                    x_streams,
                    lambda x, _i=i, _m=key_padding_mask, _r=rope_fn: self.attention_layers[_i](
                        x, attn_mask=None, key_padding_mask=_m, rotary_emb_fn=_r
                    )
                )
                x_streams = self.mhc_ffn_layers[i](
                    x_streams, lambda x, _i=i: self.forward_layers[_i](x)
                )

            # (B,L,N,C) -> weighted sum over N dim -> (B,L,C)
            final_seqs = (x_streams * self.stream_fusion.weight.view(1, 1, -1, 1)).sum(dim=2)
        else:
            x = seqs
            for i in range(len(self.attention_layers)):
                if self.use_freq and self.freq_mode == 'parallel':
                    # BSARec-faithful: _attn_fn returns combined (both branches
                    # already have internal residuals), so no outer residual here.
                    x = self._attn_fn(i, x, key_padding_mask=key_padding_mask)
                else:
                    residual = x
                    x = self._attn_fn(i, x, key_padding_mask=key_padding_mask)
                    x = residual + x
                residual = x
                x = self._ffn_fn(i, x, item_ids)
                x = residual + x
            final_seqs = x

        if self.use_freq and self.freq_mode == 'post':
            final_seqs = self._apply_freq_filter(final_seqs)

        log_feats: torch.Tensor = self.last_layernorm(final_seqs)
        return log_feats
    
    def forward(self, user_ids: Any, log_seqs: Any, pos_seqs: Any, neg_seqs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        log_feats: torch.Tensor = self.log2feats(log_seqs)

        pos_ids = torch.as_tensor(pos_seqs, dtype=torch.long, device=self.dev)
        pos_embs = F.embedding(pos_ids, self.item_emb.weight, padding_idx=0)

        neg_ids = torch.as_tensor(neg_seqs, dtype=torch.long, device=self.dev)
        neg_embs = F.embedding(neg_ids, self.item_emb.weight, padding_idx=0)

        pos_logits: torch.Tensor = (log_feats * pos_embs).sum(dim=-1)

        if neg_embs.dim() == 3:
            neg_logits: torch.Tensor = (log_feats * neg_embs).sum(dim=-1)
        else:
            neg_logits = (log_feats.unsqueeze(-2) * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, log_feats, pos_embs[:, -1, :]
    
    def predict(self, user_ids: Any, log_seqs: Any, item_indices: Any) -> torch.Tensor:
        log_feats: torch.Tensor = self.log2feats(log_seqs)
        final_feat: torch.Tensor = log_feats[:, -1, :]

        item_ids = torch.as_tensor(item_indices, dtype=torch.long, device=self.dev)
        item_embs = F.embedding(item_ids, self.item_emb.weight, padding_idx=0)
        
        logits: torch.Tensor = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits