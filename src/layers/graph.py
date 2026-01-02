"""
Graph neural network layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, in_features]
            edge_index: [2, num_edges]
            edge_weight: [num_edges] optional
            
        Returns:
            out: [num_nodes, out_features]
        """
        num_nodes = x.shape[0]
        device = x.device
        
        # Transform features
        x = self.linear(x)
        
        if edge_index.shape[1] == 0:
            return x
        
        # Aggregate neighbors
        row, col = edge_index
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=device)
        
        # Compute degree for normalization
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize edge weights
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # Message passing
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * norm.unsqueeze(-1))
        
        # Add self-loop
        out = out + x
        
        return self.dropout(out)


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        
        self.linear = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.attention = nn.Parameter(torch.Tensor(1, num_heads, 2 * out_features))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_rate)
        
        nn.init.xavier_uniform_(self.attention)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, in_features]
            edge_index: [2, num_edges]
            
        Returns:
            out: [num_nodes, out_features * num_heads] if concat else [num_nodes, out_features]
        """
        num_nodes = x.shape[0]
        
        # Linear transformation
        x = self.linear(x).view(num_nodes, self.num_heads, self.out_features)
        
        if edge_index.shape[1] == 0:
            if self.concat:
                return x.view(num_nodes, -1)
            return x.mean(dim=1)
        
        row, col = edge_index
        
        # Compute attention coefficients
        x_i = x[row]  # [num_edges, num_heads, out_features]
        x_j = x[col]  # [num_edges, num_heads, out_features]
        
        edge_features = torch.cat([x_i, x_j], dim=-1)  # [num_edges, num_heads, 2*out_features]
        
        alpha = (edge_features * self.attention).sum(dim=-1)  # [num_edges, num_heads]
        alpha = self.leaky_relu(alpha)
        
        # Softmax over neighbors
        alpha_exp = alpha.exp()
        alpha_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        alpha_sum.index_add_(0, col, alpha_exp)
        alpha_norm = alpha_exp / (alpha_sum[col] + 1e-8)
        
        alpha_norm = self.dropout(alpha_norm)
        
        # Aggregate
        out = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)
        weighted = x[row] * alpha_norm.unsqueeze(-1)
        out.index_add_(0, col, weighted)
        
        if self.concat:
            return out.view(num_nodes, -1)
        return out.mean(dim=1)


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer with mean aggregation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        
        self.linear_self = nn.Linear(in_features, out_features, bias=True)
        self.linear_neigh = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, in_features]
            edge_index: [2, num_edges]
            
        Returns:
            out: [num_nodes, out_features]
        """
        num_nodes = x.shape[0]
        device = x.device
        
        # Self transformation
        out_self = self.linear_self(x)
        
        if edge_index.shape[1] == 0:
            if self.normalize:
                return F.normalize(out_self, p=2, dim=-1)
            return out_self
        
        row, col = edge_index
        
        # Mean aggregation of neighbors
        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, col, x[row])
        
        # Count neighbors
        deg = torch.zeros(num_nodes, device=device)
        deg.index_add_(0, col, torch.ones(edge_index.shape[1], device=device))
        deg = deg.clamp(min=1)
        
        neigh_mean = neigh_sum / deg.unsqueeze(-1)
        out_neigh = self.linear_neigh(neigh_mean)
        
        # Combine
        out = out_self + out_neigh
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return self.dropout(out)
