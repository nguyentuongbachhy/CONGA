import torch
import torch.nn as nn
from typing import Optional, Tuple


class ContinuumItemEmbedding(nn.Module):
    """
    Continuum Memory System (CMS) for item embeddings.
    Implements multi-timescale learning with 3 levels:
    - Fast memory: Recent interactions (high learning rate)
    - Medium memory: Session patterns (medium learning rate)
    - Slow memory: Long-term knowledge from graph (low learning rate)
    """
    
    def __init__(
        self, 
        num_items: int, 
        embedding_dim: int,
        padding_idx: int = 0,
        fast_weight: float = 0.5,
        medium_weight: float = 0.3,
        slow_weight: float = 0.2,
        device: torch.device = torch.device('cuda')
    ):
        super(ContinuumItemEmbedding, self).__init__()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.device = device
        self.to(device)
        
        # 3-level embeddings
        self.fast_emb = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx).to(device)
        self.medium_emb = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx).to(device)
        self.slow_emb = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx).to(device)
        
        # Adaptive weights for combining embeddings
        self.register_buffer('fast_weight', torch.tensor(fast_weight, device=device))
        self.register_buffer('medium_weight', torch.tensor(medium_weight, device=device))
        self.register_buffer('slow_weight', torch.tensor(slow_weight, device=device))
        
        # Initialize all embeddings with same values
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all embeddings with Xavier uniform"""
        nn.init.xavier_uniform_(self.fast_emb.weight)
        nn.init.xavier_uniform_(self.medium_emb.weight)
        nn.init.xavier_uniform_(self.slow_emb.weight)
        
        # Zero out padding
        if self.padding_idx is not None:
            with torch.no_grad():
                self.fast_emb.weight[self.padding_idx].fill_(0)
                self.medium_emb.weight[self.padding_idx].fill_(0)
                self.slow_emb.weight[self.padding_idx].fill_(0)
    
    def init_from_pretrained(self, pretrained_embeddings: torch.Tensor):
        """
        Initialize slow memory from pretrained graph embeddings.
        Fast and medium memories are initialized randomly.
        
        Args:
            pretrained_embeddings: Tensor of shape [num_items, embedding_dim]
        """
        with torch.no_grad():
            # Slow memory gets graph knowledge
            self.slow_emb.weight.copy_(pretrained_embeddings)
            
            # Fast memory starts random (will adapt quickly)
            nn.init.xavier_uniform_(self.fast_emb.weight)
            
            # Medium memory starts as interpolation
            self.medium_emb.weight.copy_(
                0.7 * pretrained_embeddings + 0.3 * self.fast_emb.weight
            )
            
            # Zero out padding
            if self.padding_idx is not None:
                self.fast_emb.weight[self.padding_idx].fill_(0)
                self.medium_emb.weight[self.padding_idx].fill_(0)
                self.slow_emb.weight[self.padding_idx].fill_(0)
    
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Get combined embeddings from all 3 levels.
        
        Args:
            item_ids: Tensor of item indices
            
        Returns:
            Combined embeddings weighted by fast/medium/slow weights
        """
        fast = self.fast_emb(item_ids)
        medium = self.medium_emb(item_ids)
        slow = self.slow_emb(item_ids)
        
        # Weighted combination
        combined = (
            self.fast_weight * fast + 
            self.medium_weight * medium + 
            self.slow_weight * slow
        )
        
        return combined
    
    def get_parameter_groups(self, base_lr: float) -> list:
        """
        Get parameter groups with different learning rates for each level.
        
        Args:
            base_lr: Base learning rate (will be used for fast memory)
            
        Returns:
            List of parameter groups for optimizer
        """
        return [
            {
                'params': self.fast_emb.parameters(),
                'lr': base_lr,
                'name': 'fast_memory'
            },
            {
                'params': self.medium_emb.parameters(),
                'lr': base_lr * 0.1,  # 10x slower
                'name': 'medium_memory'
            },
            {
                'params': self.slow_emb.parameters(),
                'lr': base_lr * 0.01,  # 100x slower
                'name': 'slow_memory'
            }
        ]
    
    @property
    def weight(self) -> torch.Tensor:
        """
        Property to maintain compatibility with nn.Embedding interface.
        Returns the combined weight matrix.
        """
        # Create combined weight for all items
        all_ids = torch.arange(self.num_items, device=self.device)
        return self.forward(all_ids)


class ContinuumSASRec(nn.Module):
    """
    SASRec model with Continuum Memory System.
    Wrapper that replaces standard item embeddings with CMS.
    """
    
    def __init__(self, base_model: nn.Module, use_cms: bool = True):
        super(ContinuumSASRec, self).__init__()
        
        self.base_model = base_model
        self.use_cms = use_cms
        
        if use_cms:
            # Replace item_emb with ContinuumItemEmbedding
            original_emb = base_model.item_emb
            
            self.cms_emb = ContinuumItemEmbedding(
                num_items=original_emb.num_embeddings,
                embedding_dim=original_emb.embedding_dim,
                padding_idx=original_emb.padding_idx,
                device=base_model.dev
            )
            
            # Copy original weights to all levels
            with torch.no_grad():
                self.cms_emb.fast_emb.weight.copy_(original_emb.weight)
                self.cms_emb.medium_emb.weight.copy_(original_emb.weight)
                self.cms_emb.slow_emb.weight.copy_(original_emb.weight)
            
            # Replace in base model
            base_model.item_emb = self.cms_emb
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def log2feats(self, *args, **kwargs):
        return self.base_model.log2feats(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.base_model.predict(*args, **kwargs)
    
    def get_cms_parameter_groups(self, base_lr: float) -> Optional[list]:
        """Get CMS parameter groups if CMS is enabled"""
        if self.use_cms and hasattr(self.base_model.item_emb, 'get_parameter_groups'):
            return self.base_model.item_emb.get_parameter_groups(base_lr)
        return None
