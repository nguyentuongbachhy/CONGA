import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

from modules import SASRecBlock


class SASRec(nn.Module):
    def __init__(self, user_num: int, item_num: int, args: Any) -> None:
        super().__init__()
        self.item_num: int = item_num
        self.dev: str = args.device
        self.maxlen: int = args.maxlen

        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.blocks = nn.ModuleList(
            SASRecBlock(args.hidden_units, args.num_heads, args.dropout_rate)
            for _ in range(args.num_blocks)
        )
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

    def log2feats(self, seqs: torch.Tensor) -> torch.Tensor:
        """(B, T) item IDs -> (B, T, D) sequence features."""
        B, T = seqs.shape
        x = self.item_emb(seqs)  # (B, T, D)

        positions = torch.arange(1, T + 1, device=seqs.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Zero out padding tokens
        pad_mask_3d = (seqs == 0).unsqueeze(-1)   # (B, T, 1)
        x = x.masked_fill(pad_mask_3d, 0.0)

        # Build causal mask once for this sequence length
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=seqs.device), diagonal=1
        )
        key_padding_mask = seqs == 0  # (B, T)  True = ignore

        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)
            x = x.masked_fill(pad_mask_3d, 0.0)

        return self.last_layernorm(x)

    def forward(
        self,
        u: torch.Tensor,
        seq: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Training forward.

        Returns:
            pos_logits : (B, T)
            neg_logits : (B, T) or (B, T, num_neg)
            log_feats  : (B, T, D)
            None       : placeholder for API compatibility with CONGA
        """
        log_feats = self.log2feats(seq)                    # (B, T, D)
        pos_logits = (log_feats * self.item_emb(pos)).sum(-1)   # (B, T)

        if neg.dim() == 3:
            # (B, T, num_neg)
            neg_logits = (log_feats.unsqueeze(2) * self.item_emb(neg)).sum(-1)
        else:
            neg_logits = (log_feats * self.item_emb(neg)).sum(-1)  # (B, T)

        return pos_logits, neg_logits, log_feats, None

    def predict(
        self,
        u: Any,
        seq: Any,
        item_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Inference scoring compatible with utils evaluate functions.

        Args:
            u             : user IDs (unused; kept for API parity)
            seq           : (B, T) numpy array or LongTensor of item IDs
            item_indices  : (B, N) LongTensor of candidate item IDs

        Returns:
            scores : (B, N)
        """
        if not isinstance(seq, torch.Tensor):
            seq = torch.LongTensor(seq).to(self.dev)
        else:
            seq = seq.to(self.dev)

        log_feats = self.log2feats(seq)         # (B, T, D)
        final_feat = log_feats[:, -1, :]        # (B, D)

        item_embs = self.item_emb(item_indices) # (B, N, D)
        scores = (final_feat.unsqueeze(1) * item_embs).sum(-1)  # (B, N)
        return scores

