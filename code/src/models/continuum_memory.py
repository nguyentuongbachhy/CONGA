import torch
import torch.nn as nn
from typing import Optional


class ContinuumItemEmbedding(nn.Module):
    def __init__(
        self, 
        num_items: int, 
        embedding_dim: int,
        padding_idx: int = 0,
        fast_weight: float = 0.5,
        medium_weight: float = 0.3,
        slow_weight: float = 0.2,
        device: torch.device = torch.device('cuda')
    ) -> None:
        super().__init__()
        
        self.num_items: int = num_items
        self.embedding_dim: int = embedding_dim
        self.padding_idx: int = padding_idx
        self.device: torch.device = device
        self.to(device)
        
        self.fast_emb: nn.Embedding = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx).to(device)
        self.medium_emb: nn.Embedding = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx).to(device)
        self.slow_emb: nn.Embedding = nn.Embedding(num_items, embedding_dim, padding_idx=padding_idx).to(device)
        
        self.register_buffer('fast_weight', torch.tensor(fast_weight, device=device))
        self.register_buffer('medium_weight', torch.tensor(medium_weight, device=device))
        self.register_buffer('slow_weight', torch.tensor(slow_weight, device=device))
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fast_emb.weight)
        nn.init.xavier_uniform_(self.medium_emb.weight)
        nn.init.xavier_uniform_(self.slow_emb.weight)
        
        if self.padding_idx is not None:
            with torch.no_grad():
                self.fast_emb.weight[self.padding_idx].fill_(0)
                self.medium_emb.weight[self.padding_idx].fill_(0)
                self.slow_emb.weight[self.padding_idx].fill_(0)
    
    def init_from_pretrained(self, pretrained_embeddings: torch.Tensor) -> None:
        with torch.no_grad():
            self.slow_emb.weight.copy_(pretrained_embeddings)
            nn.init.xavier_uniform_(self.fast_emb.weight)
            self.medium_emb.weight.copy_(0.7 * pretrained_embeddings + 0.3 * self.fast_emb.weight)
            
            if self.padding_idx is not None:
                self.fast_emb.weight[self.padding_idx].fill_(0)
                self.medium_emb.weight[self.padding_idx].fill_(0)
                self.slow_emb.weight[self.padding_idx].fill_(0)
    
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        fast = self.fast_emb(item_ids)
        medium = self.medium_emb(item_ids)
        slow = self.slow_emb(item_ids)
        
        return self.fast_weight * fast + self.medium_weight * medium + self.slow_weight * slow
    
    def get_parameter_groups(self, base_lr: float) -> list:
        return [
            {'params': self.fast_emb.parameters(), 'lr': base_lr, 'name': 'fast_memory'},
            {'params': self.medium_emb.parameters(), 'lr': base_lr * 0.1, 'name': 'medium_memory'},
            {'params': self.slow_emb.parameters(), 'lr': base_lr * 0.01, 'name': 'slow_memory'}
        ]
    
    @property
    def weight(self) -> torch.Tensor:
        all_ids = torch.arange(self.num_items, device=self.device)
        return self.forward(all_ids)


class ContinuumSASRec(nn.Module):
    def __init__(self, base_model: nn.Module, use_cms: bool = True) -> None:
        super().__init__()
        
        self.base_model = base_model
        self.use_cms = use_cms
        
        if use_cms and hasattr(base_model, 'item_emb') and hasattr(base_model, 'dev'):
            original_emb: nn.Embedding = getattr(base_model, 'item_emb')
            dev = getattr(base_model, 'dev')
            
            self.cms_emb = ContinuumItemEmbedding(
                num_items=int(original_emb.num_embeddings),
                embedding_dim=int(original_emb.embedding_dim),
                padding_idx=int(original_emb.padding_idx) if original_emb.padding_idx is not None else 0,
                device=dev if isinstance(dev, torch.device) else torch.device(str(dev))
            )
            
            with torch.no_grad():
                self.cms_emb.fast_emb.weight.copy_(original_emb.weight)
                self.cms_emb.medium_emb.weight.copy_(original_emb.weight)
                self.cms_emb.slow_emb.weight.copy_(original_emb.weight)
            
            setattr(base_model, 'item_emb', self.cms_emb)
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def log2feats(self, *args, **kwargs):
        if hasattr(self.base_model, 'log2feats'):
            return getattr(self.base_model, 'log2feats')(*args, **kwargs)
        raise AttributeError("base_model has no log2feats method")
    
    def predict(self, *args, **kwargs):
        if hasattr(self.base_model, 'predict'):
            return getattr(self.base_model, 'predict')(*args, **kwargs)
        raise AttributeError("base_model has no predict method")
    
    def get_cms_parameter_groups(self, base_lr: float) -> Optional[list]:
        if self.use_cms and hasattr(self, 'cms_emb'):
            return self.cms_emb.get_parameter_groups(base_lr)
        return None
