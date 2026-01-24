import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class ItemCooccurrenceGraph:
    def __init__(self, window_size: int = 10, min_cooccurrence: int = 1) -> None:
        self.window_size = window_size
        self.min_cooccurrence = min_cooccurrence

    def build_from_sequences(
        self, user_sequences: Dict[int, List[int]], num_items: int, device: str = "cpu"
    ) -> torch.Tensor:
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
            return torch.sparse_coo_tensor(
                indices, values, (num_items + 1, num_items + 1), device=device
            )

        all_edges = torch.cat(all_edges, dim=1)  # [2, total_edges]
        all_weights = torch.cat(all_weights, dim=0)  # [total_edges]

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

        symmetric_indices = torch.cat([
            filtered_indices,
            filtered_indices.flip(0)
        ], dim=1)
        symmetric_values = torch.cat([filtered_values, filtered_values], dim=0)

        adj_matrix = torch.sparse_coo_tensor(
            symmetric_indices, symmetric_values, 
            (num_items + 1, num_items + 1), device=device
        ).coalesce()

        return adj_matrix

    def normalize_adjacency(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        adj_matrix = adj_matrix.coalesce()
        device = adj_matrix.device
        n_nodes = adj_matrix.size(0)

        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

        diag_indices = torch.arange(n_nodes, device=device).unsqueeze(0).repeat(2, 1)
        d_mat_inv_sqrt = torch.sparse_coo_tensor(
            diag_indices, d_inv_sqrt, (n_nodes, n_nodes), device=device
        )

        norm_adj = torch.sparse.mm(d_mat_inv_sqrt, adj_matrix)
        norm_adj = torch.sparse.mm(norm_adj, d_mat_inv_sqrt)

        return norm_adj.coalesce()


class ItemGraphEmbedding(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        device: str = "cuda",
        topk_sparsity: Optional[int] = None,
    ) -> None:
        super(ItemGraphEmbedding, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        self.topk_sparsity = topk_sparsity

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        self.item_embedding.weight.data[0] = 0

        self.Graph = None

    def set_graph(self, graph_tensor: torch.Tensor) -> None:
        self.Graph = graph_tensor.to(self.device)

        if self.topk_sparsity is not None:
            self._apply_topk_sparsity()

    def _apply_topk_sparsity(self) -> None:
        if self.Graph is None or self.topk_sparsity is None:
            return
            
        adj_dense = self.Graph.to_dense()
        n_nodes = adj_dense.size(0)
        k = self.topk_sparsity

        row_nnz = (adj_dense != 0).sum(dim=1)
        
        effective_k = torch.minimum(
            torch.full_like(row_nnz, k), row_nnz
        )

        topk_values, topk_indices = torch.topk(
            adj_dense, k, dim=1, largest=True, sorted=False
        )

        valid_mask = torch.arange(k, device=self.device).unsqueeze(0) < effective_k.unsqueeze(1)

        row_indices = torch.arange(n_nodes, device=self.device).unsqueeze(1).expand(-1, k)
        
        valid_rows = row_indices[valid_mask]
        valid_cols = topk_indices[valid_mask]
        valid_vals = topk_values[valid_mask]

        sparse_indices = torch.stack([valid_rows, valid_cols], dim=0)
        self.Graph = torch.sparse_coo_tensor(
            sparse_indices, valid_vals, (n_nodes, n_nodes), device=self.device
        ).coalesce()

    def propagate(self) -> torch.Tensor:
        if self.Graph is None:
            return self.item_embedding.weight
            
        items_emb = self.item_embedding.weight
        embs = [items_emb]

        original_dtype = items_emb.dtype

        with torch.amp.autocast_mode.autocast("cuda", enabled=False):
            items_emb = items_emb.float()
            graph_float = self.Graph.float()

            for _ in range(self.num_layers):
                items_emb = torch.sparse.mm(graph_float, items_emb)
                embs.append(items_emb)

        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)

        if original_dtype == torch.float16:
            final_emb = final_emb.half()

        return final_emb

    def forward(
        self,
        anchor_items: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_items = self.propagate()

        anchor_emb = all_items[anchor_items]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        return anchor_emb, pos_emb, neg_emb

    def get_all_embeddings(self) -> torch.Tensor:
        return self.propagate()

    def save_embeddings(self, save_path: str) -> None:
        embeddings = self.get_all_embeddings()
        torch.save(
            {
                "embeddings": embeddings.cpu(),
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
            },
            save_path,
        )
        print(f"Saved graph embeddings to {save_path}")


def bpr_loss_item_graph(
    anchor_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    reg_weight: float = 1e-4,
) -> torch.Tensor:
    pos_scores = (anchor_emb * pos_emb).sum(dim=-1)

    if neg_emb.dim() == 2:
        neg_scores = (anchor_emb * neg_emb).sum(dim=-1)
    else:
        neg_scores = (anchor_emb.unsqueeze(1) * neg_emb).sum(dim=-1)

    loss = -torch.log(
        torch.sigmoid(pos_scores.unsqueeze(-1) - neg_scores) + 1e-10
    ).mean()

    reg_loss = (
        reg_weight
        * (anchor_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2))
        / anchor_emb.size(0)
    )

    return loss + reg_loss


class ItemGraphDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        adj_matrix: torch.Tensor,
        num_items: int,
        num_negatives: int = 1,
        neg_sample_pool_size: int = 1000,
    ) -> None:
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.neg_sample_pool_size = neg_sample_pool_size

        adj_matrix = adj_matrix.coalesce()
        indices = adj_matrix.indices()
        
        mask = indices[0] < indices[1]
        mask &= (indices[0] > 0) & (indices[1] > 0)
        
        edge_i = indices[0][mask]
        edge_j = indices[1][mask]
        
        self.edges = torch.stack([edge_i, edge_j], dim=1)
        self.adj_dense = adj_matrix.to_dense() > 0

        print(f"ItemGraphDataset: {len(self.edges)} edges for training")

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int):
        anchor, pos = self.edges[idx].tolist()

        if self.num_negatives == 1:
            neg = self._sample_negative(anchor)
            return anchor, pos, neg
        else:
            negs = self._sample_negatives_batch(anchor, self.num_negatives)
            return anchor, pos, negs

    def _sample_negative(self, anchor: int) -> int:
        candidates = torch.randint(1, self.num_items + 1, (self.neg_sample_pool_size,))
        
        valid_mask = (candidates != anchor) & (~self.adj_dense[anchor, candidates])
        valid_candidates = candidates[valid_mask]
        
        if len(valid_candidates) > 0:
            return int(valid_candidates[0].item())
        else:
            neg = int(torch.randint(1, self.num_items + 1, (1,)).item())
            while neg == anchor or bool(self.adj_dense[anchor, neg].item()):
                neg = int(torch.randint(1, self.num_items + 1, (1,)).item())
            return neg

    def _sample_negatives_batch(self, anchor: int, num_neg: int) -> torch.Tensor:
        oversample_factor = 5
        num_candidates = num_neg * oversample_factor
        
        candidates = torch.randint(1, self.num_items + 1, (num_candidates,))
        
        valid_mask = (candidates != anchor) & (~self.adj_dense[anchor, candidates])
        valid_negs = candidates[valid_mask]
        
        if len(valid_negs) >= num_neg:
            return valid_negs[:num_neg]
        else:
            needed = num_neg - len(valid_negs)
            additional = []
            for _ in range(needed):
                neg = self._sample_negative(anchor)
                additional.append(neg)
            
            if len(valid_negs) > 0:
                return torch.cat([valid_negs, torch.tensor(additional, dtype=torch.long)])
            else:
                return torch.tensor(additional, dtype=torch.long)


def build_graph_from_dataset(
    dataset_name: str,
    window_size: int = 10,
    min_cooccurrence: int = 1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, int]:
    from utils import data_partition

    user_train, _, _, _, itemnum = data_partition(dataset_name)

    user_sequences = {}
    for user_id in user_train:
        seq = user_train[user_id]
        if len(seq) > 0:
            user_sequences[user_id] = seq

    graph_builder = ItemCooccurrenceGraph(
        window_size=window_size, min_cooccurrence=min_cooccurrence
    )

    adj_matrix = graph_builder.build_from_sequences(user_sequences, itemnum, device=device)

    adj_matrix = adj_matrix.coalesce()
    nnz = adj_matrix._nnz()
    
    print(f"Built item co-occurrence graph:")
    print(f"  Items: {itemnum}")
    print(f"  Edges: {nnz // 2}")
    print(f"  Avg degree: {nnz / itemnum:.2f}")

    return adj_matrix, itemnum