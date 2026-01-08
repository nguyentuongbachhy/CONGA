"""
LightGCN: Simplified Graph Convolutional Network for Recommendation
Teacher model for graph distillation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List
import scipy.sparse as sp


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        device: str = "cuda",
    ):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device

        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        nn.init.xavier_uniform_(self.user_embedding.weight[1:])
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        self.user_embedding.weight.data[0] = 0
        self.item_embedding.weight.data[0] = 0

        self.Graph = None

    def build_graph(self, user_train: Dict[int, List[int]]) -> sp.csr_matrix:
        """Build normalized adjacency matrix for user-item bipartite graph"""
        n_users = self.num_users + 1
        n_items = self.num_items + 1

        row, col, data = [], [], []
        for user_id, items in user_train.items():
            if user_id == 0:
                continue
            for item_id in items:
                if item_id == 0:
                    continue
                row.append(user_id)
                col.append(item_id)
                data.append(1.0)

        R = sp.coo_matrix(
            (data, (row, col)), shape=(n_users, n_items), dtype=np.float32
        )

        adj_mat = sp.dok_matrix(
            (n_users + n_items, n_users + n_items), dtype=np.float32
        )
        adj_mat[:n_users, n_users:] = R
        adj_mat[n_users:, :n_users] = R.T
        adj_mat = adj_mat.tocoo()

        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat @ adj_mat @ d_mat
        norm_adj = norm_adj.tocoo()

        indices = torch.LongTensor(
            np.vstack([norm_adj.row, norm_adj.col])
        ).to(self.device)
        values = torch.FloatTensor(norm_adj.data).to(self.device)
        shape = torch.Size(norm_adj.shape)

        return torch.sparse_coo_tensor(indices, values, shape, device=self.device)

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """LightGCN message passing"""
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)

        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users_out, items_out = torch.split(
            light_out, [self.num_users + 1, self.num_items + 1]
        )
        return users_out, items_out

    def forward(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            users: [batch_size]
            pos_items: [batch_size]
            neg_items: [batch_size] or [batch_size, num_neg]
        Returns:
            user_emb, pos_emb, neg_emb
        """
        all_users, all_items = self.propagate()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        if neg_items.dim() == 1:
            neg_emb = all_items[neg_items]
        else:
            neg_emb = all_items[neg_items]

        return users_emb, pos_emb, neg_emb

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings after propagation"""
        all_users, _ = self.propagate()
        return all_users[user_ids]

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get item embeddings after propagation"""
        _, all_items = self.propagate()
        return all_items[item_ids]

    def predict(self, users: torch.Tensor, seqs: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for user-item pairs
        Args:
            users: [batch_size]
            seqs: [batch_size, seq_len] (ignored for LightGCN, kept for interface compatibility)
            items: [batch_size, num_items]
        Returns:
            scores: [batch_size, num_items]
        """
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        items_emb = all_items[items]

        scores = (users_emb.unsqueeze(1) * items_emb).sum(dim=-1)
        return scores


def bpr_loss(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    reg_weight: float = 1e-4,
) -> torch.Tensor:
    """
    BPR loss for LightGCN
    Args:
        user_emb: [batch_size, dim]
        pos_emb: [batch_size, dim]
        neg_emb: [batch_size, dim] or [batch_size, num_neg, dim]
    """
    pos_scores = (user_emb * pos_emb).sum(dim=-1)

    if neg_emb.dim() == 2:
        neg_scores = (user_emb * neg_emb).sum(dim=-1)
    else:
        neg_scores = (user_emb.unsqueeze(1) * neg_emb).sum(dim=-1)

    loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(-1) - neg_scores) + 1e-10).mean()

    reg_loss = reg_weight * (
        user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)
    ) / user_emb.size(0)

    return loss + reg_loss
