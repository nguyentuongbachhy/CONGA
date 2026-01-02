"""
CONGA: COntrastive Nested Graph Architecture for Continual Sequential Recommendation

This is the main proposed model combining:
1. Nested Graph Architecture (local item-item + global user-item)
2. Multi-scale Contrastive Learning
3. Continual Learning with Memory Replay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

from .base import BaseModel
from .sasrec import SASRecBlock


class LocalItemGraph(nn.Module):
    """
    Local Item-Item Graph: captures item co-occurrence patterns within sequences.
    
    Builds a graph where edges represent sequential transitions between items.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads=4, dropout=dropout_rate, batch_first=True)
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
        """
        Apply local graph convolution on sequence items.
        
        Args:
            item_emb: [batch_size, seq_len, hidden_size]
            item_seq: [batch_size, seq_len] for masking
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # Create attention mask for valid items
        mask = (item_seq == 0)  # [B, L]
        
        hidden = item_emb
        
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            # Graph attention (all-to-all within sequence)
            attn_out, _ = gat(
                hidden, hidden, hidden,
                key_padding_mask=mask,
            )
            
            # Residual + norm
            hidden = norm(hidden + self.dropout(attn_out))
        
        return hidden


class GlobalUserItemGraph(nn.Module):
    """
    Global User-Item Graph: captures user preference patterns across all interactions.
    
    Uses a memory bank to store user representations for global context.
    """
    
    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 1000,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # User memory bank
        self.user_memory = nn.Parameter(torch.randn(memory_size, hidden_size))
        nn.init.xavier_normal_(self.user_memory)
        
        # Attention for memory retrieval
        self.memory_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, dropout=dropout_rate, batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-8)
    
    def forward(
        self,
        seq_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enhance sequence representation with global context.
        
        Args:
            seq_repr: [batch_size, hidden_size] sequence representation
            
        Returns:
            enhanced: [batch_size, hidden_size]
        """
        batch_size = seq_repr.shape[0]
        
        # Expand memory for batch
        memory = self.user_memory.unsqueeze(0).expand(batch_size, -1, -1)  # [B, M, H]
        
        # Query memory with sequence representation
        query = seq_repr.unsqueeze(1)  # [B, 1, H]
        
        global_context, _ = self.memory_attention(query, memory, memory)
        global_context = global_context.squeeze(1)  # [B, H]
        
        # Fuse local and global
        fused = torch.cat([seq_repr, global_context], dim=-1)
        enhanced = self.fusion(fused)
        enhanced = self.layer_norm(enhanced + seq_repr)  # Residual
        
        return enhanced


class NestedGraphEncoder(nn.Module):
    """
    Nested Graph Encoder: combines local and global graph representations.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_local_layers: int = 2,
        num_global_layers: int = 1,
        memory_size: int = 1000,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.local_graph = LocalItemGraph(
            hidden_size=hidden_size,
            num_layers=num_local_layers,
            dropout_rate=dropout_rate,
        )
        
        self.global_graph = GlobalUserItemGraph(
            hidden_size=hidden_size,
            memory_size=memory_size,
            num_layers=num_global_layers,
            dropout_rate=dropout_rate,
        )
    
    def forward(
        self,
        item_emb: torch.Tensor,
        item_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply nested graph encoding.
        
        Args:
            item_emb: [batch_size, seq_len, hidden_size]
            item_seq: [batch_size, seq_len]
            
        Returns:
            local_output: [batch_size, seq_len, hidden_size]
            global_output: [batch_size, hidden_size]
        """
        # Local graph encoding
        local_output = self.local_graph(item_emb, item_seq)
        
        # Get sequence representation (last valid position)
        seq_repr = local_output[:, -1, :]  # [B, H]
        
        # Global graph encoding
        global_output = self.global_graph(seq_repr)
        
        return local_output, global_output


class MultiScaleContrastive(nn.Module):
    """
    Multi-scale Contrastive Learning module.
    
    Applies contrastive learning at different scales:
    1. Item-level: within sequence
    2. Sequence-level: across sequences
    3. Graph-level: local vs global representations
    """
    
    def __init__(
        self,
        hidden_size: int,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        
        # Projection heads for different scales
        self.item_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.seq_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
    
    def sequence_contrastive_loss(
        self,
        repr_1: torch.Tensor,
        repr_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sequence-level contrastive loss.
        
        Args:
            repr_1: [batch_size, hidden_size] first view
            repr_2: [batch_size, hidden_size] second view
            
        Returns:
            loss: scalar
        """
        batch_size = repr_1.shape[0]
        
        # Project
        z1 = F.normalize(self.seq_proj(repr_1), dim=-1)
        z2 = F.normalize(self.seq_proj(repr_2), dim=-1)
        
        # Similarity matrix
        sim = torch.matmul(z1, z2.T) / self.temperature
        
        # InfoNCE loss
        labels = torch.arange(batch_size, device=repr_1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        
        return loss
    
    def graph_contrastive_loss(
        self,
        local_repr: torch.Tensor,
        global_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute local-global contrastive loss.
        
        Args:
            local_repr: [batch_size, hidden_size]
            global_repr: [batch_size, hidden_size]
            
        Returns:
            loss: scalar
        """
        batch_size = local_repr.shape[0]
        
        # Normalize
        local_z = F.normalize(local_repr, dim=-1)
        global_z = F.normalize(global_repr, dim=-1)
        
        # Similarity
        sim = torch.matmul(local_z, global_z.T) / self.temperature
        
        # Loss
        labels = torch.arange(batch_size, device=local_repr.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        
        return loss


class MemoryReplay(nn.Module):
    """
    Memory Replay module for Continual Learning.
    
    Maintains a buffer of representative samples and supports
    experience replay during training.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.max_seq_len = max_seq_len
        
        # Buffers (not trainable parameters)
        self.register_buffer("seq_buffer", torch.zeros(buffer_size, max_seq_len, dtype=torch.long))
        self.register_buffer("pos_buffer", torch.zeros(buffer_size, max_seq_len, dtype=torch.long))
        self.register_buffer("buffer_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("buffer_count", torch.zeros(1, dtype=torch.long))
    
    def add(
        self,
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
    ) -> None:
        """Add samples to buffer."""
        batch_size = item_seq.shape[0]
        
        for i in range(batch_size):
            ptr = int(self.buffer_ptr.item())
            self.seq_buffer[ptr] = item_seq[i]
            self.pos_buffer[ptr] = pos_items[i]
            
            self.buffer_ptr[0] = (ptr + 1) % self.buffer_size
            self.buffer_count[0] = min(self.buffer_count[0] + 1, self.buffer_size)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from buffer."""
        count = int(self.buffer_count.item())
        if count == 0:
            return None, None
        
        indices = torch.randint(0, count, (batch_size,))
        return self.seq_buffer[indices], self.pos_buffer[indices]


class CONGA(BaseModel):
    """
    CONGA: COntrastive Nested Graph Architecture for Continual Sequential Recommendation.
    
    Main contributions:
    1. Nested graph architecture (local + global)
    2. Multi-scale contrastive learning
    3. Continual learning with memory replay
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
        # Graph parameters
        num_local_layers: int = 2,
        num_global_layers: int = 1,
        memory_bank_size: int = 1000,
        # Contrastive parameters
        contrastive_weight: float = 0.1,
        graph_cl_weight: float = 0.1,
        temperature: float = 0.07,
        # Continual learning parameters
        use_continual: bool = False,
        replay_buffer_size: int = 10000,
        replay_ratio: float = 0.3,
        distillation_weight: float = 0.5,
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
        self.use_continual = use_continual
        self.distillation_weight = distillation_weight
        self.replay_ratio = replay_ratio
        
        # Transformer encoder (sequence modeling)
        self.encoder_blocks = nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout_rate, norm_first=True)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        
        # Nested graph encoder
        self.nested_graph = NestedGraphEncoder(
            hidden_size=hidden_size,
            num_local_layers=num_local_layers,
            num_global_layers=num_global_layers,
            memory_size=memory_bank_size,
            dropout_rate=dropout_rate,
        )
        
        # Multi-scale contrastive module
        self.contrastive = MultiScaleContrastive(
            hidden_size=hidden_size,
            temperature=temperature,
        )
        
        # Fusion layer for combining transformer and graph outputs
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fusion_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        
        # Memory replay for continual learning
        if use_continual:
            self.memory_replay = MemoryReplay(
                buffer_size=replay_buffer_size,
                max_seq_len=max_seq_len,
            )
            # Teacher model for distillation (will be set externally)
            self.teacher = None
        
        self.init_weights()
    
    def encode_sequence(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Encode sequence through transformer."""
        seq_emb = self.get_embedding(item_seq)
        attention_mask = self.get_attention_mask(item_seq.shape[1])
        
        hidden = seq_emb
        for block in self.encoder_blocks:
            hidden = block(hidden, attention_mask)
        
        return self.encoder_norm(hidden)
    
    def forward(
        self,
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        """
        batch_size = item_seq.shape[0]
        
        # Transformer encoding
        seq_output = self.encode_sequence(item_seq)  # [B, L, H]
        
        # Nested graph encoding
        local_output, global_repr = self.nested_graph(seq_output, item_seq)
        
        # Get sequence representation from transformer
        seq_repr = seq_output[:, -1, :]  # [B, H]
        local_repr = local_output[:, -1, :]  # [B, H]
        
        # Fuse transformer and graph representations
        fused = torch.cat([seq_repr, global_repr], dim=-1)
        final_repr = self.fusion(fused)
        final_repr = self.fusion_norm(final_repr + seq_repr)  # Residual
        
        # Expand for sequence-level predictions
        final_output = seq_output.clone()
        final_output[:, -1, :] = final_repr
        
        # Recommendation logits
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        
        pos_logits = (final_output * pos_emb).sum(dim=-1)
        neg_logits = (final_output * neg_emb).sum(dim=-1)
        
        # Contrastive losses
        # 1. Sequence augmentation contrastive (dropout as augmentation)
        seq_output_aug = self.encode_sequence(item_seq)
        seq_repr_aug = seq_output_aug[:, -1, :]
        seq_cl_loss = self.contrastive.sequence_contrastive_loss(seq_repr, seq_repr_aug)
        
        # 2. Local-global graph contrastive
        graph_cl_loss = self.contrastive.graph_contrastive_loss(local_repr, global_repr)
        
        outputs = {
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
        
        # Continual learning: distillation from teacher
        if self.use_continual and self.teacher is not None:
            with torch.no_grad():
                teacher_output = self.teacher.encode_sequence(item_seq)
                teacher_repr = teacher_output[:, -1, :]
            
            # Knowledge distillation loss
            distill_loss = F.mse_loss(final_repr, teacher_repr)
            outputs["distill_loss"] = distill_loss
        
        return outputs
    
    def predict(
        self,
        item_seq: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference prediction."""
        # Transformer encoding
        seq_output = self.encode_sequence(item_seq)
        
        # Nested graph encoding
        local_output, global_repr = self.nested_graph(seq_output, item_seq)
        
        seq_repr = seq_output[:, -1, :]
        
        # Fuse
        fused = torch.cat([seq_repr, global_repr], dim=-1)
        final_repr = self.fusion(fused)
        final_repr = self.fusion_norm(final_repr + seq_repr)
        
        if candidate_items is not None:
            item_emb = self.item_embedding(candidate_items)
            scores = torch.bmm(item_emb, final_repr.unsqueeze(-1)).squeeze(-1)
        else:
            # Return scores aligned with raw item ids in [0..num_items]
            # so that target_item (1..num_items) can be gathered directly.
            all_item_emb = self.item_embedding.weight  # [num_items+1, H] (include padding)
            scores = torch.matmul(final_repr, all_item_emb.T)  # [B, num_items+1]
        
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
        """Compute total CONGA loss."""
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        
        mask = (pos_items != 0).float()
        
        # Recommendation loss
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        rec_loss = (
            criterion(pos_logits, pos_labels) * mask +
            criterion(neg_logits, neg_labels) * mask
        ).sum() / mask.sum()
        
        # Contrastive losses
        seq_cl_loss = outputs["seq_cl_loss"]
        graph_cl_loss = outputs["graph_cl_loss"]
        
        total_loss = (
            rec_loss +
            self.contrastive_weight * seq_cl_loss +
            self.graph_cl_weight * graph_cl_loss
        )
        
        # Distillation loss (continual learning)
        if "distill_loss" in outputs:
            total_loss += self.distillation_weight * outputs["distill_loss"]
        
        return total_loss
    
    def update_memory(
        self,
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
    ) -> None:
        """Update memory replay buffer."""
        if self.use_continual:
            self.memory_replay.add(item_seq, pos_items)
    
    def get_replay_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get replay samples."""
        if self.use_continual:
            return self.memory_replay.sample(batch_size)
        return None, None
    
    def set_teacher(self, teacher: nn.Module) -> None:
        """Set teacher model for distillation."""
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
