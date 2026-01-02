"""
Base model class for sequential recommendation.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for sequential recommendation models.
    
    All models should implement:
    - forward: training forward pass
    - predict: inference prediction
    - get_sequence_representation: get sequence embeddings
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 64,
        max_seq_len: int = 50,
        dropout_rate: float = 0.2,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Item embedding (shared across all models)
        # padding_idx=0 for masking padding tokens
        self.item_embedding = nn.Embedding(
            num_items + 1, 
            hidden_size, 
            padding_idx=0
        )
        
        # Positional embedding
        self.position_embedding = nn.Embedding(
            max_seq_len + 1, 
            hidden_size, 
            padding_idx=0
        )
        
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-8)
    
    def get_embedding(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Get item embeddings with positional encoding.
        
        Args:
            item_seq: [batch_size, seq_len] item indices
            
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = item_seq.shape
        
        # Item embeddings
        item_emb = self.item_embedding(item_seq)  # [B, L, H]
        item_emb *= self.hidden_size ** 0.5  # Scale by sqrt(d)
        
        # Position embeddings
        positions = torch.arange(1, seq_len + 1, device=item_seq.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        positions = positions * (item_seq != 0).long()  # Mask padding positions
        pos_emb = self.position_embedding(positions)  # [B, L, H]
        
        # Combine and apply dropout
        embeddings = self.embedding_dropout(item_emb + pos_emb)
        
        return embeddings
    
    @abstractmethod
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
            pos_items: [batch_size, seq_len] positive items (next items)
            neg_items: [batch_size, seq_len] or [batch_size, seq_len, num_neg] negative samples
            
        Returns:
            Dictionary containing:
            - pos_logits: positive item scores
            - neg_logits: negative item scores
            - (optional) additional outputs for contrastive/graph losses
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        item_seq: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference prediction.
        
        Args:
            item_seq: [batch_size, seq_len] input sequence
            candidate_items: [batch_size, num_candidates] or None for all items
            
        Returns:
            scores: [batch_size, num_candidates] or [batch_size, num_items]
        """
        pass
    
    @abstractmethod
    def get_sequence_representation(
        self, 
        item_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the final sequence representation.
        
        Args:
            item_seq: [batch_size, seq_len] input sequence
            
        Returns:
            seq_repr: [batch_size, hidden_size] sequence representation
        """
        pass
    
    def get_attention_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create causal attention mask.
        
        Args:
            seq_len: sequence length
            
        Returns:
            mask: [seq_len, seq_len] causal mask (True = masked)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device),
            diagonal=1
        ).bool()
        return mask
    
    def init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            try:
                if "weight" in name and param.dim() > 1:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            except Exception:
                pass
        
        # Set padding embeddings to zero
        self.item_embedding.weight.data[0].fill_(0)
        self.position_embedding.weight.data[0].fill_(0)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
