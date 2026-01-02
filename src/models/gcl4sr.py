"""
GCL4SR: Graph Contrastive Learning for Sequential Recommendation
Paper: https://www.ijcai.org/proceedings/2022/333 (IJCAI 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

from .sasrec import SASRec, SASRecBlock


class ItemTransitionGraph(nn.Module):
    """
    Weighted Item Transition Graph construction.
    
    Builds a graph based on item co-occurrence in sequences,
    where edge weights represent transition probabilities.
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_size: int,
        num_gnn_layers: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.num_gnn_layers = num_gnn_layers
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_gnn_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size, eps=1e-8)
            for _ in range(num_gnn_layers)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
    
    def build_graph_from_batch(
        self, 
        item_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build item transition graph from batch of sequences.
        
        Args:
            item_seq: [batch_size, seq_len]
            
        Returns:
            edge_index: [2, num_edges] edge indices
            edge_weight: [num_edges] edge weights
        """
        batch_size, seq_len = item_seq.shape
        device = item_seq.device
        
        # Collect all transitions
        sources = []
        targets = []
        
        for i in range(batch_size):
            seq = item_seq[i]
            valid_mask = seq != 0
            valid_items = seq[valid_mask]
            
            if len(valid_items) < 2:
                continue
            
            # Add forward transitions
            for j in range(len(valid_items) - 1):
                sources.append(valid_items[j].item())
                targets.append(valid_items[j + 1].item())
        
        if len(sources) == 0:
            # Return empty graph
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, device=device)
            )
        
        # Convert to tensors
        sources = torch.tensor(sources, device=device)
        targets = torch.tensor(targets, device=device)
        
        # Create edge index
        edge_index = torch.stack([sources, targets], dim=0)
        
        # Compute edge weights (normalized by source node degree)
        unique_edges, counts = torch.unique(edge_index, dim=1, return_counts=True)
        edge_weight = counts.float()
        
        # Normalize weights
        edge_weight = edge_weight / edge_weight.sum()
        
        return unique_edges, edge_weight
    
    def aggregate(
        self,
        item_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simple graph aggregation.
        
        Args:
            item_emb: [num_items + 1, hidden_size]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            
        Returns:
            aggregated: [num_items + 1, hidden_size]
        """
        if edge_index.shape[1] == 0:
            return item_emb
        
        num_items = item_emb.shape[0]
        device = item_emb.device
        
        # Initialize output
        output = torch.zeros_like(item_emb)
        
        # Scatter aggregation
        sources, targets = edge_index[0], edge_index[1]
        
        # Weighted message passing
        messages = item_emb[sources] * edge_weight.unsqueeze(-1)
        output.index_add_(0, targets, messages)
        
        # Add self-loop
        output = output + item_emb
        
        return output
    
    def forward(
        self,
        item_emb: torch.Tensor,
        item_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply GNN on item embeddings.
        
        Args:
            item_emb: [num_items + 1, hidden_size] item embeddings
            item_seq: [batch_size, seq_len] for graph construction
            
        Returns:
            enhanced_emb: [num_items + 1, hidden_size]
        """
        # Build graph from batch
        edge_index, edge_weight = self.build_graph_from_batch(item_seq)
        
        hidden = item_emb
        
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Aggregate neighbors
            agg = self.aggregate(hidden, edge_index, edge_weight)
            
            # Transform
            hidden = gnn(agg)
            hidden = norm(hidden)
            hidden = self.activation(hidden)
            hidden = self.dropout(hidden)
            
            # Residual connection
            hidden = hidden + item_emb
        
        return hidden


class GCL4SR(SASRec):
    """
    GCL4SR: Graph Contrastive Learning for Sequential Recommendation.
    
    Combines:
    1. Weighted item transition graph
    2. Graph-enhanced sequence encoder
    3. Graph contrastive learning
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
        # GCL4SR specific
        num_gnn_layers: int = 2,
        contrastive_weight: float = 0.1,
        temperature: float = 0.2,
        graph_dropout: float = 0.1,
    ):
        super().__init__(
            num_items=num_items,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            norm_first=norm_first,
            device=device,
        )
        
        self.num_gnn_layers = num_gnn_layers
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.graph_dropout = graph_dropout
        
        # Item transition graph module
        self.item_graph = ItemTransitionGraph(
            num_items=num_items,
            hidden_size=hidden_size,
            num_gnn_layers=num_gnn_layers,
            dropout_rate=graph_dropout,
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
    
    def encode_sequence_with_graph(
        self,
        item_seq: torch.Tensor,
        use_graph: bool = True,
    ) -> torch.Tensor:
        """
        Encode sequence with optional graph enhancement.
        
        Args:
            item_seq: [batch_size, seq_len]
            use_graph: whether to use graph-enhanced embeddings
            
        Returns:
            seq_output: [batch_size, seq_len, hidden_size]
        """
        if use_graph:
            # Get graph-enhanced item embeddings
            enhanced_emb = self.item_graph(
                self.item_embedding.weight,
                item_seq,
            )
            
            # Use enhanced embeddings
            batch_size, seq_len = item_seq.shape
            seq_emb = enhanced_emb[item_seq]  # [B, L, H]
            seq_emb *= self.hidden_size ** 0.5
            
            # Add positional embeddings
            positions = torch.arange(1, seq_len + 1, device=item_seq.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            positions = positions * (item_seq != 0).long()
            pos_emb = self.position_embedding(positions)
            
            seq_emb = self.embedding_dropout(seq_emb + pos_emb)
        else:
            seq_emb = self.get_embedding(item_seq)
        
        # Create causal attention mask
        attention_mask = self.get_attention_mask(seq_emb.shape[1])
        
        # Pass through transformer blocks
        hidden = seq_emb
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        
        output = self.final_norm(hidden)
        
        return output
    
    def forward(
        self,
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with graph contrastive learning.
        """
        batch_size = item_seq.shape[0]
        
        # Encode with graph enhancement
        seq_output_graph = self.encode_sequence_with_graph(item_seq, use_graph=True)
        
        # Encode without graph (for contrastive)
        seq_output_seq = self.encode_sequence(item_seq)
        
        # Get final representations
        graph_repr = seq_output_graph[:, -1, :]  # [B, H]
        seq_repr = seq_output_seq[:, -1, :]  # [B, H]
        
        # Project for contrastive learning
        graph_proj = self.projection(graph_repr)
        seq_proj = self.projection(seq_repr)
        
        # Normalize
        graph_proj = F.normalize(graph_proj, dim=-1)
        seq_proj = F.normalize(seq_proj, dim=-1)
        
        # Graph-Sequence contrastive loss
        sim_matrix = torch.matmul(graph_proj, seq_proj.T) / self.temperature
        labels = torch.arange(batch_size, device=item_seq.device)
        
        cl_loss = (
            F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)
        ) / 2
        
        # Recommendation loss (use graph-enhanced output)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        
        pos_logits = (seq_output_graph * pos_emb).sum(dim=-1)
        neg_logits = (seq_output_graph * neg_emb).sum(dim=-1)
        
        return {
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
            "seq_output": seq_output_graph,
            "cl_loss": cl_loss,
            "graph_repr": graph_repr,
            "seq_repr": seq_repr,
        }
    
    def predict(
        self,
        item_seq: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference with graph-enhanced encoding."""
        seq_output = self.encode_sequence_with_graph(item_seq, use_graph=True)
        final_repr = seq_output[:, -1, :]
        
        if candidate_items is not None:
            item_emb = self.item_embedding(candidate_items)
            scores = torch.bmm(item_emb, final_repr.unsqueeze(-1)).squeeze(-1)
        else:
            # Return scores aligned with raw item ids in [0..num_items]
            # so that target_item (1..num_items) can be gathered directly.
            all_item_emb = self.item_embedding.weight  # [num_items+1, H] (include padding)
            scores = torch.matmul(final_repr, all_item_emb.T)  # [B, num_items+1]
        
        return scores
    
    def compute_total_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        pos_items: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute total loss including graph contrastive component."""
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        cl_loss = outputs["cl_loss"]
        
        mask = (pos_items != 0).float()
        
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        rec_loss = (
            criterion(pos_logits, pos_labels) * mask +
            criterion(neg_logits, neg_labels) * mask
        ).sum() / mask.sum()
        
        total_loss = rec_loss + self.contrastive_weight * cl_loss
        
        return total_loss
