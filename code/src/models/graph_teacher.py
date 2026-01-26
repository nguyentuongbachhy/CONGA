import torch
import torch.nn as nn
from typing import Tuple, Dict, List


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

    def build_graph(self, user_train: Dict[int, List[int]]) -> torch.Tensor:
        n_users = self.num_users + 1
        n_items = self.num_items + 1

        user_indices = []
        item_indices = []
        
        for user_id, items in user_train.items():
            if user_id == 0:
                continue
            for item_id in items:
                if item_id == 0:
                    continue
                user_indices.append(user_id)
                item_indices.append(item_id)

        edge_index_user = torch.LongTensor(user_indices).to(self.device)
        edge_index_item = torch.LongTensor(item_indices).to(self.device)
        edge_values = torch.ones(len(user_indices), dtype=torch.float32).to(self.device)

        # adj_mat = [[0, R], [R^T, 0]]
        total_nodes = n_users + n_items
        
        # Top-right block: user -> item
        row_top = edge_index_user
        col_top = edge_index_item + n_users
        
        # Bottom-left block: item -> user
        row_bottom = edge_index_item + n_users
        col_bottom = edge_index_user
        
        # Combine both directions
        row = torch.cat([row_top, row_bottom])
        col = torch.cat([col_top, col_bottom])
        values = torch.cat([edge_values, edge_values])
        
        # Create sparse adjacency matrix
        indices = torch.stack([row, col], dim=0)
        adj_mat = torch.sparse_coo_tensor(
            indices, values, (total_nodes, total_nodes), device=self.device
        )
        
        # Compute degree normalization: D^(-1/2) @ A @ D^(-1/2)
        adj_mat = adj_mat.coalesce()
        row_sum = torch.sparse.sum(adj_mat, dim=1).to_dense()
        
        # D^(-1/2)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        
        # Create D^(-1/2) as diagonal sparse matrix
        diag_indices = torch.arange(total_nodes, device=self.device).unsqueeze(0).repeat(2, 1)
        d_mat_inv_sqrt = torch.sparse_coo_tensor(
            diag_indices, d_inv_sqrt, (total_nodes, total_nodes), device=self.device
        )
        
        # Normalize: D^(-1/2) @ A @ D^(-1/2)
        norm_adj = torch.sparse.mm(d_mat_inv_sqrt, adj_mat)
        norm_adj = torch.sparse.mm(norm_adj, d_mat_inv_sqrt)
        
        return norm_adj.coalesce()

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
        all_users, all_items = self.propagate()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        if neg_items.dim() == 1:
            neg_emb = all_items[neg_items]
        else:
            neg_emb = all_items[neg_items]

        return users_emb, pos_emb, neg_emb

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        all_users, _ = self.propagate()
        return all_users[user_ids]

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        _, all_items = self.propagate()
        return all_items[item_ids]

    def predict(self, users: torch.Tensor, seqs: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
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
