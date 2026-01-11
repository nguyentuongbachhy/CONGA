import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sp
from collections import defaultdict
from pathlib import Path


class ItemCooccurrenceGraph:
    """Build item co-occurrence graph from user interaction sequences."""

    def __init__(self, window_size: int = 10, min_cooccurrence: int = 1):
        self.window_size = window_size
        self.min_cooccurrence = min_cooccurrence

    def build_from_sequences(
        self, user_sequences: Dict[int, List[int]], num_items: int
    ) -> sp.csr_matrix:
        """
        Build item-item co-occurrence graph from user interaction sequences.

        Args:
            user_sequences: Dict mapping user_id -> list of item_ids (chronological)
            num_items: Total number of items

        Returns:
            Sparse adjacency matrix (num_items+1, num_items+1)
        """
        cooccurrence = defaultdict(int)

        for user_id, items in user_sequences.items():
            if len(items) < 2:
                continue

            for i in range(len(items)):
                item_i = items[i]
                if item_i == 0:
                    continue

                window_start = max(0, i - self.window_size)
                window_end = min(len(items), i + self.window_size + 1)

                for j in range(window_start, window_end):
                    if i == j:
                        continue
                    item_j = items[j]
                    if item_j == 0:
                        continue

                    distance = abs(i - j)
                    weight = 1.0 / distance

                    edge = tuple(sorted([item_i, item_j]))
                    cooccurrence[edge] += weight

        row, col, data = [], [], []
        for (item_i, item_j), weight in cooccurrence.items():
            if weight >= self.min_cooccurrence:
                row.extend([item_i, item_j])
                col.extend([item_j, item_i])
                data.extend([weight, weight])

        adj_matrix = sp.coo_matrix(
            (data, (row, col)), shape=(num_items + 1, num_items + 1), dtype=np.float32
        )

        return adj_matrix.tocsr()

    def normalize_adjacency(self, adj_matrix: sp.csr_matrix) -> torch.Tensor:
        """
        Normalize adjacency matrix: D^{-1/2} A D^{-1/2}

        Args:
            adj_matrix: Sparse adjacency matrix

        Returns:
            Normalized sparse tensor
        """
        adj_coo = adj_matrix.tocoo()

        rowsum = np.array(adj_matrix.sum(axis=1)).flatten()
        # Avoid divide by zero warning
        rowsum[rowsum == 0] = 1.0
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)

        norm_adj = d_mat @ adj_coo @ d_mat
        norm_adj = norm_adj.tocoo()

        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)

        return torch.sparse_coo_tensor(indices, values, shape)


class ItemGraphEmbedding(nn.Module):
    """LightGCN-style graph embedding for items only."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        device: str = "cuda",
        topk_sparsity: Optional[int] = None,
    ):
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

    def set_graph(self, graph_tensor: torch.Tensor):
        """Set the normalized adjacency matrix."""
        self.Graph = graph_tensor.to(self.device)

        if self.topk_sparsity is not None:
            self._apply_topk_sparsity()

    def _apply_topk_sparsity(self):
        """Keep only top-k neighbors for each item to reduce memory."""
        adj_dense = self.Graph.to_dense()

        for i in range(adj_dense.size(0)):
            row = adj_dense[i]
            if row.sum() == 0:
                continue
            topk_vals, topk_idx = torch.topk(
                row, min(self.topk_sparsity, row.nonzero().size(0))
            )
            mask = torch.zeros_like(row, dtype=torch.bool)
            mask[topk_idx] = True
            adj_dense[i] = row * mask.float()

        self.Graph = adj_dense.to_sparse_coo()

    def propagate(self) -> torch.Tensor:
        """
        Graph convolution propagation.

        Returns:
            Item embeddings after graph convolution
        """
        items_emb = self.item_embedding.weight
        embs = [items_emb]

        original_dtype = items_emb.dtype

        with torch.amp.autocast_mode.autocast("cuda", enabled=False):
            items_emb = items_emb.float()
            self.Graph = self.Graph.float()

            for _ in range(self.num_layers):
                items_emb = torch.sparse.mm(self.Graph, items_emb)
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
        """
        Forward pass for BPR loss.

        Args:
            anchor_items: Anchor item IDs [batch_size]
            pos_items: Positive item IDs [batch_size]
            neg_items: Negative item IDs [batch_size] or [batch_size, num_neg]

        Returns:
            Tuple of (anchor_emb, pos_emb, neg_emb)
        """
        all_items = self.propagate()

        anchor_emb = all_items[anchor_items]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        return anchor_emb, pos_emb, neg_emb

    def get_all_embeddings(self) -> torch.Tensor:
        """Get all item embeddings after propagation."""
        return self.propagate()

    def save_embeddings(self, save_path: str):
        """Save pretrained embeddings to file."""
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
    """
    BPR loss for item graph embeddings.

    Args:
        anchor_emb: Anchor item embeddings [batch_size, dim]
        pos_emb: Positive item embeddings [batch_size, dim]
        neg_emb: Negative item embeddings [batch_size, dim] or [batch_size, num_neg, dim]
        reg_weight: L2 regularization weight

    Returns:
        BPR loss scalar
    """
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
    """Dataset for training item graph embeddings."""

    def __init__(
        self,
        adj_matrix: sp.csr_matrix,
        num_items: int,
        num_negatives: int = 1,
    ):
        self.adj_matrix = adj_matrix
        self.num_items = num_items
        self.num_negatives = num_negatives

        self.edges = []
        adj_coo = adj_matrix.tocoo()
        for i, j in zip(adj_coo.row, adj_coo.col):
            if i < j and i > 0 and j > 0:
                self.edges.append((i, j))

        print(f"ItemGraphDataset: {len(self.edges)} edges for training")

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int):
        anchor, pos = self.edges[idx]

        neg_samples = []
        for _ in range(self.num_negatives):
            neg = np.random.randint(1, self.num_items + 1)
            while neg == anchor or neg == pos or self.adj_matrix[anchor, neg] > 0:
                neg = np.random.randint(1, self.num_items + 1)
            neg_samples.append(neg)

        if self.num_negatives == 1:
            return anchor, pos, neg_samples[0]
        else:
            return anchor, pos, np.array(neg_samples, dtype=np.int64)


def build_graph_from_dataset(
    dataset_name: str,
    window_size: int = 10,
    min_cooccurrence: int = 1,
) -> Tuple[sp.csr_matrix, int]:
    """
    Build item co-occurrence graph from dataset.

    Args:
        dataset_name: Name of dataset (e.g., 'ml-1m')
        window_size: Window size for co-occurrence
        min_cooccurrence: Minimum co-occurrence count

    Returns:
        Tuple of (adjacency_matrix, num_items)
    """
    from utils import data_partition

    user_train, user_valid, user_test, usernum, itemnum = data_partition(dataset_name)

    user_sequences = {}
    for user_id in user_train:
        seq = user_train[user_id]
        if len(seq) > 0:
            user_sequences[user_id] = seq

    graph_builder = ItemCooccurrenceGraph(
        window_size=window_size, min_cooccurrence=min_cooccurrence
    )

    adj_matrix = graph_builder.build_from_sequences(user_sequences, itemnum)

    print(f"Built item co-occurrence graph:")
    print(f"  Items: {itemnum}")
    print(f"  Edges: {adj_matrix.nnz // 2}")
    print(f"  Avg degree: {adj_matrix.nnz / itemnum:.2f}")

    return adj_matrix, itemnum
