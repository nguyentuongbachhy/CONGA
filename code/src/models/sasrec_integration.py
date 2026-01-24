import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Any


def load_graph_embeddings(
    embedding_path: str,
    expected_num_items: Optional[int] = None,
    expected_dim: Optional[int] = None,
) -> torch.Tensor:
    if not Path(embedding_path).exists():
        raise FileNotFoundError(f"Graph embeddings not found at {embedding_path}")
    
    checkpoint = torch.load(embedding_path, map_location='cpu')
    embeddings = checkpoint['embeddings']
    num_items = checkpoint['num_items']
    embedding_dim = checkpoint['embedding_dim']
    
    if expected_num_items is not None and num_items != expected_num_items:
        raise ValueError(f"Embedding num_items mismatch: expected {expected_num_items}, got {num_items}")
    
    if expected_dim is not None and embedding_dim != expected_dim:
        raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {embedding_dim}")
    
    print(f"Loaded graph embeddings: {embeddings.shape}")
    return embeddings


def initialize_sasrec_with_graph_embeddings(
    sasrec_model: nn.Module,
    graph_embedding_path: str,
    freeze_embeddings: bool = False,
    scale_factor: float = 1.0,
) -> None:
    num_items: int = getattr(sasrec_model, 'item_num')
    item_emb: nn.Embedding = getattr(sasrec_model, 'item_emb')
    embedding_dim: int = int(item_emb.embedding_dim)
    
    graph_embeddings = load_graph_embeddings(
        graph_embedding_path,
        expected_num_items=num_items,
        expected_dim=embedding_dim,
    )
    
    with torch.no_grad():
        item_emb.weight.data.copy_(graph_embeddings * scale_factor)
        item_emb.weight.data[0, :] = 0
    
    if freeze_embeddings:
        item_emb.weight.requires_grad = False
        print("Item embeddings frozen")
    else:
        print("Item embeddings initialized from graph")
    
    print(f"✓ SASRec initialized with graph embeddings (scale={scale_factor})")


def create_sasrec_with_graph_init(
    user_num: int,
    item_num: int,
    args: Any,
    graph_embedding_path: Optional[str] = None,
    freeze_embeddings: bool = False,
    scale_factor: float = 1.0,
) -> nn.Module:
    from model import SASRec
    
    model = SASRec(user_num, item_num, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    
    if graph_embedding_path is not None and Path(graph_embedding_path).exists():
        initialize_sasrec_with_graph_embeddings(
            model,
            graph_embedding_path,
            freeze_embeddings=freeze_embeddings,
            scale_factor=scale_factor,
        )
    else:
        model.item_emb.weight.data[0, :] = 0
        print("SASRec initialized with random embeddings")
    
    return model


class GraphEnhancedSASRec(nn.Module):
    graph_embeddings: torch.Tensor
    
    def __init__(
        self,
        sasrec_model: nn.Module,
        graph_embedding_path: str,
        fusion_mode: str = "add",
        fusion_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.sasrec = sasrec_model
        self.fusion_mode = fusion_mode
        self.fusion_weight = fusion_weight
        
        item_num = getattr(sasrec_model, 'item_num')
        item_emb: nn.Embedding = getattr(sasrec_model, 'item_emb')
        emb_dim = int(item_emb.embedding_dim)
        
        graph_embeddings = load_graph_embeddings(
            graph_embedding_path,
            expected_num_items=item_num,
            expected_dim=emb_dim,
        )
        
        self.register_buffer('graph_embeddings', graph_embeddings)
        
        if fusion_mode == "gate":
            self.fusion_gate = nn.Linear(emb_dim * 2, emb_dim)
        
        print(f"GraphEnhancedSASRec: fusion_mode={fusion_mode}, weight={fusion_weight}")
    
    def get_fused_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_emb: nn.Embedding = getattr(self.sasrec, 'item_emb')
        learned_emb = item_emb(item_ids)
        graph_emb = self.graph_embeddings[item_ids].to(learned_emb.device)
        
        if self.fusion_mode == "add":
            return learned_emb + self.fusion_weight * graph_emb
        elif self.fusion_mode == "gate":
            concat = torch.cat([learned_emb, graph_emb], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(concat))
            return gate * learned_emb + (1 - gate) * graph_emb
        else:
            return learned_emb
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        item_emb: nn.Embedding = getattr(self.sasrec, 'item_emb')
        original_emb = item_emb
        emb_dim = int(item_emb.embedding_dim)
        
        class FusedEmbedding(nn.Module):
            def __init__(self, parent: 'GraphEnhancedSASRec', dim: int) -> None:
                super().__init__()
                self.parent = parent
                self.embedding_dim = dim
            
            def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
                return self.parent.get_fused_embeddings(item_ids)
        
        setattr(self.sasrec, 'item_emb', FusedEmbedding(self, emb_dim))
        output = self.sasrec(user_ids, log_seqs, pos_seqs, neg_seqs)
        setattr(self.sasrec, 'item_emb', original_emb)
        
        return output
    
    def predict(self, user_ids, log_seqs, item_indices):
        item_emb: nn.Embedding = getattr(self.sasrec, 'item_emb')
        original_emb = item_emb
        emb_dim = int(item_emb.embedding_dim)
        
        class FusedEmbedding(nn.Module):
            def __init__(self, parent: 'GraphEnhancedSASRec', dim: int) -> None:
                super().__init__()
                self.parent = parent
                self.embedding_dim = dim
            
            def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
                return self.parent.get_fused_embeddings(item_ids)
        
        setattr(self.sasrec, 'item_emb', FusedEmbedding(self, emb_dim))
        
        if hasattr(self.sasrec, 'predict'):
            output = getattr(self.sasrec, 'predict')(user_ids, log_seqs, item_indices)
        else:
            raise AttributeError("sasrec has no predict method")
        
        setattr(self.sasrec, 'item_emb', original_emb)
        
        return output


def compare_embeddings(
    random_embedding_path: str,
    graph_embedding_path: str,
    num_samples: int = 100,
) -> None:
    random_emb = torch.load(random_embedding_path, map_location='cpu')['embeddings']
    graph_emb = torch.load(graph_embedding_path, map_location='cpu')['embeddings']
    
    num_items = min(random_emb.size(0), graph_emb.size(0))
    sample_ids = np.random.choice(num_items, min(num_samples, num_items), replace=False)
    
    random_sample = random_emb[sample_ids]
    graph_sample = graph_emb[sample_ids]
    
    random_sim = torch.mm(random_sample, random_sample.t())
    graph_sim = torch.mm(graph_sample, graph_sample.t())
    
    random_std = random_sim[torch.triu(torch.ones_like(random_sim), diagonal=1) == 1].std()
    graph_std = graph_sim[torch.triu(torch.ones_like(graph_sim), diagonal=1) == 1].std()
    
    print("\n=== Embedding Quality Comparison ===")
    print(f"Random embeddings - Similarity std: {random_std:.4f}")
    print(f"Graph embeddings  - Similarity std: {graph_std:.4f}")
    print(f"Graph embeddings have {graph_std/random_std:.2f}x more structure")
