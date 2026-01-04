"""
SASRec-RoPE: SASRec with Rotary Position Embedding

Enhancements over base SASRec:
1. Rotary Position Embedding (RoPE) instead of learned positional encoding
2. GELU activation in FFN
3. Custom attention mechanism with RoPE integration
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable

from .base import BaseModel
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class PointWiseFeedForwardGELU(nn.Module):
    """Point-wise feed-forward network with GELU activation."""
    
    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.transpose(-1, -2)
        out = self.dropout1(self.gelu(self.conv1(out)))
        out = self.dropout2(self.conv2(out))
        return out.transpose(-1, -2)


class RoPEAttention(nn.Module):
    """Multi-head attention with RoPE integration."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_emb_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: [seq_len, seq_len] causal mask
            rotary_emb_fn: Function to get RoPE embeddings
        """
        B, L, H = x.shape
        
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim)
        
        if rotary_emb_fn is not None:
            cos, sin = rotary_emb_fn(L, x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, H)
        
        output = self.out_proj(context)
        return output


class SASRecRoPEBlock(nn.Module):
    """Transformer block with RoPE and GELU."""
    
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
        self.attention = RoPEAttention(hidden_size, num_heads, dropout_rate)
        
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ffn = PointWiseFeedForwardGELU(hidden_size, dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_emb_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: [seq_len, seq_len] causal mask
            rotary_emb_fn: RoPE function
        """
        if self.norm_first:
            normed = self.attention_norm(x)
            attn_out = self.attention(normed, attention_mask, rotary_emb_fn)
            x = x + attn_out
            x = x + self.ffn(self.ffn_norm(x))
        else:
            attn_out = self.attention(x, attention_mask, rotary_emb_fn)
            x = self.attention_norm(x + attn_out)
            x = self.ffn_norm(x + self.ffn(x))
        
        return x


class SASRecRoPE(BaseModel):
    """
    SASRec with Rotary Position Embedding.
    
    Key improvements:
    - RoPE for better position encoding
    - GELU activation for smoother gradients
    - Custom attention with RoPE integration
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
        
        head_dim = hidden_size // num_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)
        
        self.blocks = nn.ModuleList([
            SASRecRoPEBlock(hidden_size, num_heads, dropout_rate, norm_first)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        
        self.init_weights()
    
    def get_embedding(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Get item embeddings WITHOUT positional encoding.
        RoPE will be applied in attention layers.
        """
        seq_emb = self.item_embedding(item_seq)
        seq_emb = seq_emb * (self.hidden_size ** 0.5)
        seq_emb = self.embedding_dropout(seq_emb)
        return seq_emb
    
    def encode_sequence(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence through transformer blocks with RoPE.
        
        Args:
            item_seq: [batch_size, seq_len]
            
        Returns:
            seq_repr: [batch_size, seq_len, hidden_size]
        """
        seq_emb = self.get_embedding(item_seq)
        
        seq_len = item_seq.shape[1]
        attention_mask = self.get_attention_mask(seq_len)
        
        hidden = seq_emb
        for block in self.blocks:
            hidden = block(hidden, attention_mask, rotary_emb_fn=self.rope)
        
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
        seq_output = self.encode_sequence(item_seq)
        
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        
        pos_logits = (seq_output * pos_emb).sum(dim=-1)
        neg_logits = (seq_output * neg_emb).sum(dim=-1)
        
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
        seq_output = self.encode_sequence(item_seq)
        final_repr = seq_output[:, -1, :]
        
        if candidate_items is not None:
            item_emb = self.item_embedding(candidate_items)
            scores = torch.bmm(item_emb, final_repr.unsqueeze(-1)).squeeze(-1)
        else:
            all_item_emb = self.item_embedding.weight
            scores = torch.matmul(final_repr, all_item_emb.T)
        
        return scores
    
    def get_sequence_representation(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Get final sequence representation."""
        seq_output = self.encode_sequence(item_seq)
        return seq_output[:, -1, :]
