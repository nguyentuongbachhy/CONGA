import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class EnrichedGraphBuilder:
    """Build enriched graph with co-occurrence + meta-paths (item-user-item)"""
    
    def __init__(self, window_size: int = 10, min_cooccurrence: int = 1):
        self.window_size = window_size
        self.min_cooccurrence = min_cooccurrence
    
    def build_from_sequences(
        self, 
        user_sequences: Dict[int, List[int]], 
        num_users: int,
        num_items: int, 
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build enriched graph with:
        1. Item-item co-occurrence edges
        2. Item-user-item meta-path edges
        
        Returns:
            item_item_graph: Item co-occurrence adjacency matrix
            user_item_graph: Bipartite user-item graph for meta-paths
        """
        # 1. Build item co-occurrence graph (same as before)
        all_edges = []
        all_weights = []
        
        for _, items in user_sequences.items():
            if len(items) < 2:
                continue
            
            item_tensor = torch.tensor(items, dtype=torch.long, device=device)
            valid_mask = item_tensor > 0
            valid_items = item_tensor[valid_mask]
            valid_positions = torch.arange(len(items), device=device)[valid_mask]
            
            if len(valid_items) < 2:
                continue
            
            pos_diff = valid_positions.unsqueeze(1) - valid_positions.unsqueeze(0)
            window_mask = (pos_diff.abs() <= self.window_size) & (pos_diff != 0)
            edge_i, edge_j = window_mask.nonzero(as_tuple=True)
            
            if len(edge_i) == 0:
                continue
            
            items_i = valid_items[edge_i]
            items_j = valid_items[edge_j]
            distances = pos_diff[edge_i, edge_j].abs().float()
            weights = 1.0 / distances
            
            edge_min = torch.minimum(items_i, items_j)
            edge_max = torch.maximum(items_i, items_j)
            
            all_edges.append(torch.stack([edge_min, edge_max], dim=0))
            all_weights.append(weights)
        
        if len(all_edges) == 0:
            indices = torch.zeros((2, 0), dtype=torch.long, device=device)
            values = torch.zeros(0, dtype=torch.float32, device=device)
            item_item_graph = torch.sparse_coo_tensor(
                indices, values, (num_items + 1, num_items + 1), device=device
            )
        else:
            all_edges = torch.cat(all_edges, dim=1)
            all_weights = torch.cat(all_weights, dim=0)
            
            temp_sparse = torch.sparse_coo_tensor(
                all_edges, all_weights, (num_items + 1, num_items + 1), device=device
            ).coalesce()
            
            if self.min_cooccurrence > 1:
                mask = temp_sparse.values() >= self.min_cooccurrence
                filtered_indices = temp_sparse.indices()[:, mask]
                filtered_values = temp_sparse.values()[mask]
            else:
                filtered_indices = temp_sparse.indices()
                filtered_values = temp_sparse.values()
            
            symmetric_indices = torch.cat([filtered_indices, filtered_indices.flip(0)], dim=1)
            symmetric_values = torch.cat([filtered_values, filtered_values], dim=0)
            
            item_item_graph = torch.sparse_coo_tensor(
                symmetric_indices, symmetric_values, 
                (num_items + 1, num_items + 1), device=device
            ).coalesce()
        
        # 2. Build user-item bipartite graph for meta-paths
        user_indices = []
        item_indices = []
        
        for user_id, items in user_sequences.items():
            for item_id in items:
                if item_id > 0:
                    user_indices.append(user_id)
                    item_indices.append(item_id)
        
        if len(user_indices) > 0:
            edge_indices = torch.tensor([user_indices, item_indices], dtype=torch.long, device=device)
            edge_values = torch.ones(len(user_indices), dtype=torch.float32, device=device)
            user_item_graph = torch.sparse_coo_tensor(
                edge_indices, edge_values, (num_users + 1, num_items + 1), device=device
            ).coalesce()
        else:
            indices = torch.zeros((2, 0), dtype=torch.long, device=device)
            values = torch.zeros(0, dtype=torch.float32, device=device)
            user_item_graph = torch.sparse_coo_tensor(
                indices, values, (num_users + 1, num_items + 1), device=device
            )
        
        return item_item_graph, user_item_graph


class GATAttention(nn.Module):
    """GAT-style attention for NGCF propagation"""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 2 * out_dim))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h: [N, in_dim]
        adj: [N, N] sparse adjacency
        """
        Wh = self.W(h)
        
        # Compute attention coefficients
        edge_index = adj.coalesce().indices()
        if edge_index.size(1) == 0:
            return Wh
        
        h_i = Wh[edge_index[0]]
        h_j = Wh[edge_index[1]]
        
        # Concatenate for attention
        a_input = torch.cat([h_i, h_j], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a.t()).squeeze())
        
        # Sparse attention
        attention = torch.sparse_coo_tensor(
            edge_index, e, adj.size(), device=h.device
        ).coalesce()
        
        # Softmax per node
        attention_values = torch.sparse.softmax(attention, dim=1)
        attention = attention_values.coalesce()
        
        # Apply dropout (only during training)
        if self.training and self.dropout.p > 0:
            mask = torch.bernoulli(torch.ones_like(attention.values()) * (1 - self.dropout.p))
            attention = torch.sparse_coo_tensor(
                attention.indices(), attention.values() * mask, attention.size(), device=h.device
            ).coalesce()
        
        # Message passing
        return torch.sparse.mm(attention, Wh)


class BiInteractionAggregator(nn.Module):
    """KGLN-style Bi-Interaction aggregator with personalization"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
    
    def forward(self, ego_emb: torch.Tensor, neighbor_emb: torch.Tensor) -> torch.Tensor:
        """
        Bi-interaction: combines element-wise product and sum
        ego_emb: [N, dim]
        neighbor_emb: [N, dim] aggregated neighbor embeddings
        """
        # Element-wise interaction (like NGCF)
        interaction = ego_emb * neighbor_emb
        
        # Transform both branches
        sum_emb = self.W1(ego_emb + neighbor_emb)
        prod_emb = self.W2(interaction)
        
        return sum_emb + prod_emb


class MultiScaleAttention(nn.Module):
    """GR-MC style multi-scale attention with edge dropout for continual learning"""
    
    def __init__(self, dim: int, num_scales: int = 3, edge_dropout: float = 0.1):
        super().__init__()
        self.num_scales = num_scales
        self.edge_dropout = edge_dropout
        
        # Multi-scale attention weights
        self.scale_attns = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(num_scales)
        ])
        self.scale_combine = nn.Linear(num_scales * dim, dim)
        
        for attn in self.scale_attns:
            nn.init.xavier_uniform_(attn.weight)
        nn.init.xavier_uniform_(self.scale_combine.weight)
    
    def apply_edge_dropout(self, adj: torch.Tensor, dropout_rate: float) -> torch.Tensor:
        """Apply edge dropout for robustness"""
        if not self.training or dropout_rate == 0:
            return adj
        
        adj = adj.coalesce()
        edge_mask = torch.bernoulli(torch.ones_like(adj.values()) * (1 - dropout_rate))
        
        return torch.sparse_coo_tensor(
            adj.indices(), adj.values() * edge_mask, adj.size(), device=adj.device
        ).coalesce()
    
    def forward(self, h: torch.Tensor, adj_list: List[torch.Tensor]) -> torch.Tensor:
        """
        h: [N, dim]
        adj_list: List of adjacency matrices at different scales
        """
        scale_outputs = []
        
        # Process only first few scales for efficiency
        max_scales = min(self.num_scales, len(adj_list))
        
        for i in range(max_scales):
            adj = adj_list[i]
            
            # Apply edge dropout
            adj_dropped = self.apply_edge_dropout(adj, self.edge_dropout)
            
            # Scale-specific transformation
            h_scale = self.scale_attns[i](h)
            
            # Aggregate with dropped edges
            if adj_dropped._nnz() > 0:
                h_agg = torch.sparse.mm(adj_dropped, h_scale)
            else:
                h_agg = h_scale
            
            scale_outputs.append(h_agg)
        
        # Combine multi-scale representations
        if len(scale_outputs) > 1:
            combined = torch.cat(scale_outputs, dim=-1)
            return self.scale_combine(combined)
        else:
            return scale_outputs[0]


class KGLNEncoder(nn.Module):
    """
    KGLN backbone with:
    - Bi-Interaction aggregators
    - GAT attention for propagation
    - Multi-scale attention (GR-MC)
    - User-relation influence factors
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        use_gat: bool = True,
        use_multiscale: bool = True,
        edge_dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_gat = use_gat
        self.use_multiscale = use_multiscale
        self.device = device
        
        # Embeddings
        self.user_emb = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # User-relation influence factors (KGLN personalization)
        self.user_relation_factors = nn.Parameter(torch.ones(num_users + 1, num_layers))
        
        # Aggregators
        self.bi_aggregators = nn.ModuleList([
            BiInteractionAggregator(embedding_dim) for _ in range(num_layers)
        ])
        
        if use_gat:
            self.gat_layers = nn.ModuleList([
                GATAttention(embedding_dim, embedding_dim) for _ in range(num_layers)
            ])
        
        if use_multiscale:
            self.multiscale_attn = MultiScaleAttention(
                embedding_dim, num_scales=min(3, num_layers), edge_dropout=edge_dropout
            )
        
        self.reset_parameters()
        
        # Graph placeholders
        self.item_item_graph = None
        self.user_item_graph = None
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight[1:])
        nn.init.xavier_uniform_(self.item_emb.weight[1:])
        self.user_emb.weight.data[0] = 0
        self.item_emb.weight.data[0] = 0
        nn.init.ones_(self.user_relation_factors)
    
    def set_graphs(self, item_item_graph: torch.Tensor, user_item_graph: torch.Tensor):
        """Set the enriched graphs"""
        self.item_item_graph = item_item_graph.to(self.device)
        self.user_item_graph = user_item_graph.to(self.device)
        
        # Normalize graphs (user-item graph is bipartite)
        self.item_item_graph = self._normalize_adjacency(self.item_item_graph, is_bipartite=False)
        self.user_item_graph = self._normalize_adjacency(self.user_item_graph, is_bipartite=True)
    
    def _normalize_adjacency(self, adj: torch.Tensor, is_bipartite: bool = False) -> torch.Tensor:
        """Normalize adjacency matrix with D^-0.5 * A * D^-0.5 (CPU-based to avoid OOM)"""
        adj = adj.coalesce()
        original_device = adj.device
        
        # Move to CPU for normalization to avoid CUDA OOM
        adj_cpu = adj.cpu()
        
        if is_bipartite:
            # For bipartite graphs (user-item), normalize by row and column separately
            n_rows, n_cols = adj_cpu.size()
            
            # Row normalization (users)
            row_sum = torch.sparse.sum(adj_cpu, dim=1).to_dense()
            row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
            d_inv_sqrt_row = torch.pow(row_sum, -0.5)
            d_inv_sqrt_row[torch.isinf(d_inv_sqrt_row)] = 0.0
            
            # Column normalization (items)
            col_sum = torch.sparse.sum(adj_cpu, dim=0).to_dense()
            col_sum = torch.where(col_sum == 0, torch.ones_like(col_sum), col_sum)
            d_inv_sqrt_col = torch.pow(col_sum, -0.5)
            d_inv_sqrt_col[torch.isinf(d_inv_sqrt_col)] = 0.0
            
            # Apply normalization: D_row^-0.5 * A * D_col^-0.5
            indices = adj_cpu.indices()
            values = adj_cpu.values()
            
            row_indices = indices[0]
            col_indices = indices[1]
            
            normalized_values = values * d_inv_sqrt_row[row_indices] * d_inv_sqrt_col[col_indices]
            
            norm_adj = torch.sparse_coo_tensor(
                indices, normalized_values, adj_cpu.size()
            )
        else:
            # For square graphs (item-item), use element-wise normalization to avoid sparse mm
            n_rows = adj_cpu.size(0)
            
            row_sum = torch.sparse.sum(adj_cpu, dim=1).to_dense()
            row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
            d_inv_sqrt = torch.pow(row_sum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            
            # Element-wise normalization instead of matrix multiplication
            indices = adj_cpu.indices()
            values = adj_cpu.values()
            
            row_indices = indices[0]
            col_indices = indices[1]
            
            # D^-0.5 * A * D^-0.5 = values * d_inv_sqrt[row] * d_inv_sqrt[col]
            normalized_values = values * d_inv_sqrt[row_indices] * d_inv_sqrt[col_indices]
            
            norm_adj = torch.sparse_coo_tensor(
                indices, normalized_values, adj_cpu.size()
            )
        
        # Move back to original device
        norm_adj = norm_adj.coalesce()
        if original_device.type == 'cuda':
            norm_adj = norm_adj.to(original_device)
        
        return norm_adj
    
    def compute_metapath_graph(self) -> Optional[torch.Tensor]:
        """Compute item-user-item meta-path graph (memory-efficient)"""
        if self.user_item_graph is None:
            return None
        
        # Skip meta-path computation for GPU memory constraints
        # Instead, use user-item graph directly in multi-scale attention
        if self.user_item_graph.size(0) > 2000:  # Skip for large graphs
            return None
        
        # I-U-I: User-item^T * User-item (only for smaller graphs)
        ui_t = self.user_item_graph.t().coalesce()
        metapath = torch.sparse.mm(ui_t, self.user_item_graph)
        
        # Remove self-loops
        metapath = metapath.coalesce()
        indices = metapath.indices()
        values = metapath.values()
        mask = indices[0] != indices[1]
        
        return torch.sparse_coo_tensor(
            indices[:, mask], values[mask], metapath.size(), device=self.device
        ).coalesce()
    
    def propagate(self, user_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-layer graph propagation with KGLN/GAT/GR-MC enhancements
        
        Returns:
            user_embs: [num_users+1, dim] if user_ids is None, else [batch, dim]
            item_embs: [num_items+1, dim]
        """
        if self.item_item_graph is None:
            return self.user_emb.weight, self.item_emb.weight
        
        all_item_embs = [self.item_emb.weight]
        
        # Compute meta-path graph (skip for large graphs to save memory)
        metapath_graph = self.compute_metapath_graph()
        
        # Build multi-scale adjacency list
        adj_scales = [self.item_item_graph]
        if metapath_graph is not None:
            adj_scales.append(self._normalize_adjacency(metapath_graph, is_bipartite=False))
        
        current_item_emb = self.item_emb.weight
        
        with torch.amp.autocast_mode.autocast("cuda", enabled=False):
            current_item_emb = current_item_emb.float()
            
            for layer in range(self.num_layers):
                # Aggregate neighbors
                if self.use_gat and hasattr(self, 'gat_layers'):
                    neighbor_emb = self.gat_layers[layer](current_item_emb, self.item_item_graph.float())
                else:
                    neighbor_emb = torch.sparse.mm(self.item_item_graph.float(), current_item_emb)
                
                # Bi-Interaction aggregation
                current_item_emb = self.bi_aggregators[layer](current_item_emb, neighbor_emb)
                
                # Multi-scale attention (only if we have multiple scales)
                if self.use_multiscale and len(adj_scales) > 1:
                    multiscale_emb = self.multiscale_attn(current_item_emb, 
                                                          [adj.float() for adj in adj_scales])
                    current_item_emb = current_item_emb + multiscale_emb
                
                current_item_emb = F.relu(current_item_emb)
                all_item_embs.append(current_item_emb)
        
        # Mean pooling across layers
        final_item_emb = torch.mean(torch.stack(all_item_embs, dim=1), dim=1)
        
        # User embeddings (personalized with relation factors)
        if user_ids is not None:
            user_base_emb = self.user_emb(user_ids)
            # Apply user-relation influence
            user_factors = self.user_relation_factors[user_ids].mean(dim=1, keepdim=True)
            final_user_emb = user_base_emb * user_factors
        else:
            final_user_emb = self.user_emb.weight
        
        return final_user_emb, final_item_emb
    
    def forward(
        self,
        anchor_items: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for BPR training"""
        _, all_item_embs = self.propagate(user_ids)
        
        anchor_emb = all_item_embs[anchor_items]
        pos_emb = all_item_embs[pos_items]
        neg_emb = all_item_embs[neg_items]
        
        return anchor_emb, pos_emb, neg_emb
    
    def get_all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get final user and item embeddings"""
        return self.propagate()


def contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss (CONGA style)"""
    # Normalize embeddings
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    negatives = F.normalize(negatives, p=2, dim=-1)
    
    # Positive similarity
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature
    
    # Negative similarities
    if negatives.dim() == 2:
        neg_sim = torch.matmul(anchor, negatives.t()) / temperature
    else:
        neg_sim = torch.sum(anchor.unsqueeze(1) * negatives, dim=-1) / temperature
    
    # InfoNCE loss
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    return F.cross_entropy(logits, labels)


def bpr_contrastive_loss(
    anchor_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    bpr_weight: float = 0.5,
    contrastive_weight: float = 0.5,
    reg_weight: float = 1e-4,
) -> torch.Tensor:
    """Combined BPR + Contrastive loss"""
    # BPR loss
    pos_scores = (anchor_emb * pos_emb).sum(dim=-1)
    
    if neg_emb.dim() == 2:
        neg_scores = (anchor_emb * neg_emb).sum(dim=-1)
    else:
        neg_scores = (anchor_emb.unsqueeze(1) * neg_emb).sum(dim=-1)
    
    bpr_loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(-1) - neg_scores) + 1e-10).mean()
    
    # Contrastive loss
    cl_loss = contrastive_loss(anchor_emb, pos_emb, neg_emb)
    
    # Regularization
    reg_loss = reg_weight * (
        anchor_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
    ) / anchor_emb.size(0)
    
    return bpr_weight * bpr_loss + contrastive_weight * cl_loss + reg_loss
