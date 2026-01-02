"""
SASRec: Self-Attentive Sequential Recommendation
Paper: https://arxiv.org/abs/1808.09781
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .base import BaseModel


class PointWiseFeedForward(nn.Module):
    """Point-wise feed-forward network."""
    
    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H] -> transpose -> [B, H, L]
        out = x.transpose(-1, -2)
        out = self.dropout1(self.relu(self.conv1(out)))
        out = self.dropout2(self.conv2(out))
        return out.transpose(-1, -2)


class SASRecBlock(nn.Module):
    """Single transformer block for SASRec."""
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        dropout_rate: float,
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first
        
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=dropout_rate,
            batch_first=False,  # PyTorch default
        )
        
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ffn = PointWiseFeedForward(hidden_size, dropout_rate)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: [seq_len, seq_len] causal mask
        """
        # Transpose for MultiheadAttention: [B, L, H] -> [L, B, H]
        x = x.transpose(0, 1)
        
        if self.norm_first:
            # Pre-norm architecture (better for training)
            normed = self.attention_norm(x)
            attn_out, _ = self.attention(normed, normed, normed, attn_mask=attention_mask)
            x = x + attn_out
            x = x.transpose(0, 1)  # [L, B, H] -> [B, L, H]
            x = x + self.ffn(self.ffn_norm(x))
        else:
            # Post-norm architecture (original transformer)
            attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
            x = self.attention_norm(x + attn_out)
            x = x.transpose(0, 1)  # [L, B, H] -> [B, L, H]
            x = self.ffn_norm(x + self.ffn(x))
        
        return x


class SASRec(BaseModel):
    """
    SASRec: Self-Attentive Sequential Recommendation.
    
    Uses self-attention mechanism to capture sequential patterns.
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 64,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
        norm_first: bool = True,
        device: str = "cuda",
    ):
        super().__init__(
            num_items=num_items,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate,
            device=device,
        )
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm_first = norm_first
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout_rate, norm_first)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        
        self.init_weights()
    
    def encode_sequence(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence through transformer blocks.
        
        Args:
            item_seq: [batch_size, seq_len]
            
        Returns:
            seq_repr: [batch_size, seq_len, hidden_size]
        """
        # Get embeddings with positional encoding
        seq_emb = self.get_embedding(item_seq)  # [B, L, H]
        
        # Create causal attention mask
        seq_len = item_seq.shape[1]
        attention_mask = self.get_attention_mask(seq_len)
        
        # Pass through transformer blocks
        hidden = seq_emb
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        
        # Final layer norm
        output = self.final_norm(hidden)
        
        return output
    
    def forward(
        self, 
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            item_seq: [batch_size, seq_len] input sequence
            pos_items: [batch_size, seq_len] positive items
            neg_items: [batch_size, seq_len] negative items
            
        Returns:
            Dictionary with pos_logits and neg_logits
        """
        # Encode sequence
        seq_output = self.encode_sequence(item_seq)  # [B, L, H]
        
        # Get item embeddings
        pos_emb = self.item_embedding(pos_items)  # [B, L, H]
        neg_emb = self.item_embedding(neg_items)  # [B, L, H]
        
        # Compute logits via dot product
        pos_logits = (seq_output * pos_emb).sum(dim=-1)  # [B, L]
        neg_logits = (seq_output * neg_emb).sum(dim=-1)  # [B, L]
        
        return {
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
            "seq_output": seq_output,
        }
    
    def predict(
        self, 
        item_seq: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference prediction.
        
        Args:
            item_seq: [batch_size, seq_len]
            candidate_items: [batch_size, num_candidates] or None
            
        Returns:
            scores: [batch_size, num_candidates] or [batch_size, num_items]
        """
        # Get final sequence representation
        seq_output = self.encode_sequence(item_seq)  # [B, L, H]
        final_repr = seq_output[:, -1, :]  # [B, H] - use last position
        
        if candidate_items is not None:
            # Score specific candidate items
            item_emb = self.item_embedding(candidate_items)  # [B, C, H]
            scores = torch.bmm(item_emb, final_repr.unsqueeze(-1)).squeeze(-1)  # [B, C]
        else:
            # Score all items
            # Return scores aligned with raw item ids in [0..num_items]
            # so that target_item (1..num_items) can be gathered directly.
            all_item_emb = self.item_embedding.weight  # [num_items+1, H] (include padding)
            scores = torch.matmul(final_repr, all_item_emb.T)  # [B, num_items+1]
        
        return scores
    
    def get_sequence_representation(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Get final sequence representation."""
        seq_output = self.encode_sequence(item_seq)
        return seq_output[:, -1, :]  # [B, H]
