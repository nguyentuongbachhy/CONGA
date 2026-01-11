import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Any


def load_graph_embeddings(
    embedding_path: str,
    expected_num_items: Optional[int] = None,
    expected_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Load pretrained graph embeddings from file.
    
    Args:
        embedding_path: Path to saved embeddings (.pt file)
        expected_num_items: Expected number of items (for validation)
        expected_dim: Expected embedding dimension (for validation)
        
    Returns:
        Embeddings tensor [num_items+1, embedding_dim]
    """
    if not Path(embedding_path).exists():
        raise FileNotFoundError(f"Graph embeddings not found at {embedding_path}")
    
    checkpoint = torch.load(embedding_path, map_location='cpu')
    embeddings = checkpoint['embeddings']
    num_items = checkpoint['num_items']
    embedding_dim = checkpoint['embedding_dim']
    
    if expected_num_items is not None and num_items != expected_num_items:
        raise ValueError(
            f"Embedding num_items mismatch: expected {expected_num_items}, got {num_items}"
        )
    
    if expected_dim is not None and embedding_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {embedding_dim}"
        )
    
    print(f"Loaded graph embeddings: {embeddings.shape}")
    return embeddings


def initialize_sasrec_with_graph_embeddings(
    sasrec_model: nn.Module,
    graph_embedding_path: str,
    freeze_embeddings: bool = False,
    scale_factor: float = 1.0,
) -> None:
    """
    Initialize SASRec item embeddings with pretrained graph embeddings.
    
    Args:
        sasrec_model: SASRec model instance
        graph_embedding_path: Path to pretrained graph embeddings
        freeze_embeddings: Whether to freeze embeddings during training
        scale_factor: Scale factor for embeddings (default: 1.0)
    """
    num_items = sasrec_model.item_num
    embedding_dim = sasrec_model.item_emb.embedding_dim
    
    graph_embeddings = load_graph_embeddings(
        graph_embedding_path,
        expected_num_items=num_items,
        expected_dim=embedding_dim,
    )
    
    with torch.no_grad():
        sasrec_model.item_emb.weight.data.copy_(graph_embeddings * scale_factor)
        sasrec_model.item_emb.weight.data[0, :] = 0
    
    if freeze_embeddings:
        sasrec_model.item_emb.weight.requires_grad = False
        print("Item embeddings frozen (will not be updated during training)")
    else:
        print("Item embeddings initialized from graph (will be fine-tuned)")
    
    print(f"✓ SASRec initialized with graph embeddings (scale={scale_factor})")


def create_sasrec_with_graph_init(
    user_num: int,
    item_num: int,
    args: Any,
    graph_embedding_path: Optional[str] = None,
    freeze_embeddings: bool = False,
    scale_factor: float = 1.0,
) -> nn.Module:
    """
    Create SASRec model with optional graph embedding initialization.
    
    Args:
        user_num: Number of users
        item_num: Number of items
        args: Arguments object with model hyperparameters
        graph_embedding_path: Path to pretrained graph embeddings (optional)
        freeze_embeddings: Whether to freeze embeddings
        scale_factor: Scale factor for graph embeddings (0.0-1.0)
        
    Returns:
        SASRec model instance
    """
    from model import SASRec
    
    model = SASRec(user_num, item_num, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
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
        print("SASRec initialized with random embeddings (no graph init)")
    
    return model


class GraphEnhancedSASRec(nn.Module):
    """
    SASRec with graph-enhanced item embeddings.
    Combines learned embeddings with graph embeddings.
    """
    
    def __init__(
        self,
        sasrec_model: nn.Module,
        graph_embedding_path: str,
        fusion_mode: str = "add",
        fusion_weight: float = 0.5,
    ):
        super(GraphEnhancedSASRec, self).__init__()
        self.sasrec = sasrec_model
        self.fusion_mode = fusion_mode
        self.fusion_weight = fusion_weight
        
        graph_embeddings = load_graph_embeddings(
            graph_embedding_path,
            expected_num_items=sasrec_model.item_num,
            expected_dim=sasrec_model.item_emb.embedding_dim,
        )
        
        self.register_buffer('graph_embeddings', graph_embeddings)
        
        if fusion_mode == "gate":
            self.fusion_gate = nn.Linear(
                sasrec_model.item_emb.embedding_dim * 2,
                sasrec_model.item_emb.embedding_dim
            )
        
        print(f"GraphEnhancedSASRec: fusion_mode={fusion_mode}, weight={fusion_weight}")
    
    def get_fused_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Fuse learned and graph embeddings."""
        learned_emb = self.sasrec.item_emb(item_ids)
        graph_emb = self.graph_embeddings[item_ids].to(learned_emb.device)
        
        if self.fusion_mode == "add":
            return learned_emb + self.fusion_weight * graph_emb
        elif self.fusion_mode == "concat":
            raise NotImplementedError("Concat fusion requires dimension adjustment")
        elif self.fusion_mode == "gate":
            concat = torch.cat([learned_emb, graph_emb], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(concat))
            return gate * learned_emb + (1 - gate) * graph_emb
        else:
            return learned_emb
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """Forward pass with fused embeddings."""
        original_item_emb = self.sasrec.item_emb
        
        class FusedEmbedding(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
                self.embedding_dim = original_item_emb.embedding_dim
            
            def forward(self, item_ids):
                return self.parent.get_fused_embeddings(item_ids)
        
        self.sasrec.item_emb = FusedEmbedding(self)
        output = self.sasrec(user_ids, log_seqs, pos_seqs, neg_seqs)
        self.sasrec.item_emb = original_item_emb
        
        return output
    
    def predict(self, user_ids, log_seqs, item_indices):
        """Prediction with fused embeddings."""
        original_item_emb = self.sasrec.item_emb
        
        class FusedEmbedding(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
                self.embedding_dim = original_item_emb.embedding_dim
            
            def forward(self, item_ids):
                return self.parent.get_fused_embeddings(item_ids)
        
        self.sasrec.item_emb = FusedEmbedding(self)
        output = self.sasrec.predict(user_ids, log_seqs, item_indices)
        self.sasrec.item_emb = original_item_emb
        
        return output


def compare_embeddings(
    random_embedding_path: str,
    graph_embedding_path: str,
    num_samples: int = 100,
) -> None:
    """
    Compare random vs graph embeddings quality.
    
    Args:
        random_embedding_path: Path to randomly initialized embeddings
        graph_embedding_path: Path to graph embeddings
        num_samples: Number of items to sample for comparison
    """
    import numpy as np
    
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
