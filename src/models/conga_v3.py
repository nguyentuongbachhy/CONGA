"""
CONGA v3: Enhanced CONGA with RoPE and modern improvements.

Improvements over v2:
1. RoPE (Rotary Position Embedding) for better position encoding
2. GELU activation throughout (FFN, fusion layers)
3. Custom attention with RoPE integration
4. Optimized architecture for single-task training
5. Better gradient flow and training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable
import math

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


class TransformerBlockRoPE(nn.Module):
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


class LocalItemGraph(nn.Module):
    """Local Item-Item Graph with RoPE-enhanced attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        head_dim = hidden_size // num_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=512)
        
        self.gat_layers = nn.ModuleList([
            RoPEAttention(hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size, eps=1e-8)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        item_emb: torch.Tensor,
        item_seq: torch.Tensor,
    ) -> torch.Tensor:
        mask = (item_seq == 0)
        seq_len = item_emb.shape[1]
        attention_mask = torch.ones((seq_len, seq_len), device=item_emb.device)
        
        hidden = item_emb
        
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            attn_out = gat(hidden, attention_mask, rotary_emb_fn=self.rope)
            hidden = norm(hidden + self.dropout(attn_out))
        
        return hidden


class GlobalUserItemGraph(nn.Module):
    """Global User-Item Graph with GELU activation."""
    
    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 1000,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        self.user_memory = nn.Parameter(torch.randn(memory_size, hidden_size))
        nn.init.xavier_normal_(self.user_memory)
        
        self.memory_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_rate, batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-8)
    
    def forward(
        self,
        seq_repr: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = seq_repr.shape[0]
        
        memory = self.user_memory.unsqueeze(0).expand(batch_size, -1, -1)
        query = seq_repr.unsqueeze(1)
        
        global_context, _ = self.memory_attention(query, memory, memory)
        global_context = global_context.squeeze(1)
        
        fused = torch.cat([seq_repr, global_context], dim=-1)
        enhanced = self.fusion(fused)
        enhanced = self.layer_norm(enhanced + seq_repr)
        
        return enhanced


class NestedGraphEncoder(nn.Module):
    """Nested Graph Encoder with RoPE."""
    
    def __init__(
        self,
        hidden_size: int,
        num_local_layers: int = 2,
        num_global_layers: int = 1,
        memory_size: int = 1000,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.local_graph = LocalItemGraph(
            hidden_size=hidden_size,
            num_layers=num_local_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        
        self.global_graph = GlobalUserItemGraph(
            hidden_size=hidden_size,
            memory_size=memory_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
    
    def forward(
        self,
        item_emb: torch.Tensor,
        item_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_output = self.local_graph(item_emb, item_seq)
        seq_repr = local_output[:, -1, :]
        global_output = self.global_graph(seq_repr)
        
        return local_output, global_output


class MultiScaleContrastive(nn.Module):
    """Multi-scale Contrastive Learning with GELU."""
    
    def __init__(
        self,
        hidden_size: int,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        
        self.item_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.seq_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
    
    def sequence_contrastive_loss(
        self,
        repr_1: torch.Tensor,
        repr_2: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = repr_1.shape[0]
        
        z1 = F.normalize(self.seq_proj(repr_1), dim=-1)
        z2 = F.normalize(self.seq_proj(repr_2), dim=-1)
        
        sim = torch.matmul(z1, z2.T) / self.temperature
        
        labels = torch.arange(batch_size, device=repr_1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        
        return loss
    
    def graph_contrastive_loss(
        self,
        local_repr: torch.Tensor,
        global_repr: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = local_repr.shape[0]
        
        local_z = F.normalize(local_repr, dim=-1)
        global_z = F.normalize(global_repr, dim=-1)
        
        sim = torch.matmul(local_z, global_z.T) / self.temperature
        
        labels = torch.arange(batch_size, device=local_repr.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        
        return loss


class CONGAv3(BaseModel):
    """
    CONGA v3: Enhanced with RoPE and modern improvements.
    
    Key improvements over v2:
    - RoPE for better position encoding
    - GELU activation throughout
    - Custom attention with RoPE
    - Optimized for single-task training
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 64,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        device: str = "cuda",
        num_local_layers: int = 2,
        num_global_layers: int = 1,
        memory_bank_size: int = 1000,
        contrastive_weight: float = 0.1,
        graph_cl_weight: float = 0.1,
        temperature: float = 0.07,
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
        self.contrastive_weight = contrastive_weight
        self.graph_cl_weight = graph_cl_weight
        self.temperature = temperature
        
        head_dim = hidden_size // num_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlockRoPE(hidden_size, num_heads, dropout_rate, norm_first=True)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        
        self.nested_graph = NestedGraphEncoder(
            hidden_size=hidden_size,
            num_local_layers=num_local_layers,
            num_global_layers=num_global_layers,
            memory_size=memory_bank_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        
        self.contrastive = MultiScaleContrastive(
            hidden_size=hidden_size,
            temperature=temperature,
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fusion_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        
        self.init_weights()
    
    def get_embedding(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Get item embeddings WITHOUT positional encoding (RoPE handles this)."""
        seq_emb = self.item_embedding(item_seq)
        seq_emb = seq_emb * (self.hidden_size ** 0.5)
        seq_emb = self.embedding_dropout(seq_emb)
        return seq_emb
    
    def encode_sequence(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Encode sequence through transformer with RoPE."""
        seq_emb = self.get_embedding(item_seq)
        attention_mask = self.get_attention_mask(item_seq.shape[1])
        
        hidden = seq_emb
        for block in self.encoder_blocks:
            hidden = block(hidden, attention_mask, rotary_emb_fn=self.rope)
        
        return self.encoder_norm(hidden)
    
    def forward(
        self,
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass."""
        seq_output = self.encode_sequence(item_seq)
        
        local_output, global_repr = self.nested_graph(seq_output, item_seq)
        
        seq_repr = seq_output[:, -1, :]
        local_repr = local_output[:, -1, :]
        
        fused = torch.cat([seq_repr, global_repr], dim=-1)
        final_repr = self.fusion(fused)
        final_repr = self.fusion_norm(final_repr + seq_repr)
        
        final_output = seq_output.clone()
        final_output[:, -1, :] = final_repr
        
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        
        pos_logits = (final_output * pos_emb).sum(dim=-1)
        neg_logits = (final_output * neg_emb).sum(dim=-1)
        
        seq_output_aug = self.encode_sequence(item_seq)
        seq_repr_aug = seq_output_aug[:, -1, :]
        seq_cl_loss = self.contrastive.sequence_contrastive_loss(seq_repr, seq_repr_aug)
        
        graph_cl_loss = self.contrastive.graph_contrastive_loss(local_repr, global_repr)
        
        return {
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
            "seq_output": final_output,
            "seq_cl_loss": seq_cl_loss,
            "graph_cl_loss": graph_cl_loss,
            "seq_repr": seq_repr,
            "local_repr": local_repr,
            "global_repr": global_repr,
            "final_repr": final_repr,
        }
    
    def predict(
        self,
        item_seq: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference prediction."""
        seq_output = self.encode_sequence(item_seq)
        local_output, global_repr = self.nested_graph(seq_output, item_seq)
        seq_repr = seq_output[:, -1, :]
        
        fused = torch.cat([seq_repr, global_repr], dim=-1)
        final_repr = self.fusion(fused)
        final_repr = self.fusion_norm(final_repr + seq_repr)
        
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
        local_output, global_repr = self.nested_graph(seq_output, item_seq)
        seq_repr = seq_output[:, -1, :]
        
        fused = torch.cat([seq_repr, global_repr], dim=-1)
        final_repr = self.fusion(fused)
        final_repr = self.fusion_norm(final_repr + seq_repr)
        
        return final_repr
    
    def compute_total_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        pos_items: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute total CONGA v3 loss."""
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        
        mask = (pos_items != 0).float()
        
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        rec_loss = (
            criterion(pos_logits, pos_labels) * mask +
            criterion(neg_logits, neg_labels) * mask
        ).sum() / mask.sum()
        
        seq_cl_loss = outputs["seq_cl_loss"]
        graph_cl_loss = outputs["graph_cl_loss"]
        
        total_loss = (
            rec_loss +
            self.contrastive_weight * seq_cl_loss +
            self.graph_cl_weight * graph_cl_loss
        )
        
        return total_loss
