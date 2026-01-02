"""
Graph construction utilities for GNN-based sequential recommendation.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from scipy.sparse import csr_matrix


class GraphBuilder:
    """
    Builds various types of graphs for sequential recommendation.
    
    Supports:
    1. Item Transition Graph: edges between consecutive items
    2. Item Co-occurrence Graph: edges between items in same sequence
    3. User-Item Graph: bipartite graph of users and items
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        window_size: int = 5,
    ):
        """
        Args:
            num_users: Number of users
            num_items: Number of items
            window_size: Window size for co-occurrence
        """
        self.num_users = num_users
        self.num_items = num_items
        self.window_size = window_size
    
    def build_transition_graph(
        self,
        sequences: Dict[int, List[int]],
        directed: bool = True,
        weighted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build item transition graph from sequences.
        
        Args:
            sequences: Dict of user_id -> item list
            directed: Whether edges are directed
            weighted: Whether to compute edge weights
            
        Returns:
            edge_index: [2, num_edges]
            edge_weight: [num_edges] (or None if not weighted)
        """
        edge_counts = defaultdict(int)
        
        for user_id, items in sequences.items():
            for i in range(len(items) - 1):
                src, dst = items[i], items[i + 1]
                edge_counts[(src, dst)] += 1
                
                if not directed:
                    edge_counts[(dst, src)] += 1
        
        if not edge_counts:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0) if weighted else None
            )
        
        edges = list(edge_counts.keys())
        sources = [e[0] for e in edges]
        targets = [e[1] for e in edges]
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        
        if weighted:
            weights = torch.tensor([edge_counts[e] for e in edges], dtype=torch.float)
            # Normalize by source node degree
            src_degrees = defaultdict(int)
            for (s, t), c in edge_counts.items():
                src_degrees[s] += c
            weights = weights / torch.tensor([src_degrees[s] for s in sources], dtype=torch.float)
            return edge_index, weights
        
        return edge_index, None
    
    def build_cooccurrence_graph(
        self,
        sequences: Dict[int, List[int]],
        window_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build item co-occurrence graph.
        
        Args:
            sequences: Dict of user_id -> item list
            window_size: Window for co-occurrence (None = entire sequence)
            
        Returns:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        """
        if window_size is None:
            window_size = self.window_size
        
        edge_counts = defaultdict(int)
        
        for user_id, items in sequences.items():
            for i, item_i in enumerate(items):
                # Look at items within window
                window_start = max(0, i - window_size)
                window_end = min(len(items), i + window_size + 1)
                
                for j in range(window_start, window_end):
                    if i == j:
                        continue
                    
                    item_j = items[j]
                    if item_i != item_j:
                        edge_counts[(item_i, item_j)] += 1
        
        if not edge_counts:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0)
            )
        
        edges = list(edge_counts.keys())
        sources = [e[0] for e in edges]
        targets = [e[1] for e in edges]
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        weights = torch.tensor([edge_counts[e] for e in edges], dtype=torch.float)
        
        # Normalize
        weights = weights / weights.max()
        
        return edge_index, weights
    
    def build_user_item_graph(
        self,
        sequences: Dict[int, List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build user-item bipartite graph.
        
        Node indexing:
        - Users: 0 to num_users - 1
        - Items: num_users to num_users + num_items - 1
        
        Args:
            sequences: Dict of user_id -> item list
            
        Returns:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        """
        sources = []
        targets = []
        weights = []
        
        for user_id, items in sequences.items():
            item_counts = defaultdict(int)
            for item in items:
                item_counts[item] += 1
            
            for item, count in item_counts.items():
                # User -> Item edges
                sources.append(user_id)
                targets.append(self.num_users + item)
                weights.append(count)
                
                # Item -> User edges (for message passing)
                sources.append(self.num_users + item)
                targets.append(user_id)
                weights.append(count)
        
        if not sources:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0)
            )
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        
        # Normalize
        edge_weight = edge_weight / edge_weight.max()
        
        return edge_index, edge_weight
    
    def build_session_graph(
        self,
        item_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build session graph from a batch of sequences.
        Each sequence is treated as a separate graph.
        
        Args:
            item_seq: [batch_size, seq_len]
            
        Returns:
            edge_index: [2, num_edges] with batch info
            edge_weight: [num_edges]
        """
        batch_size, seq_len = item_seq.shape
        device = item_seq.device
        
        all_sources = []
        all_targets = []
        all_weights = []
        
        for b in range(batch_size):
            seq = item_seq[b]
            valid_mask = seq != 0
            valid_items = seq[valid_mask]
            
            if len(valid_items) < 2:
                continue
            
            # Forward edges
            for i in range(len(valid_items) - 1):
                all_sources.append(valid_items[i].item())
                all_targets.append(valid_items[i + 1].item())
                all_weights.append(1.0)
        
        if not all_sources:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, device=device)
            )
        
        # Aggregate duplicate edges
        edge_dict = defaultdict(float)
        for s, t, w in zip(all_sources, all_targets, all_weights):
            edge_dict[(s, t)] += w
        
        sources = [e[0] for e in edge_dict.keys()]
        targets = [e[1] for e in edge_dict.keys()]
        weights = list(edge_dict.values())
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
        edge_weight = torch.tensor(weights, dtype=torch.float, device=device)
        
        # Normalize
        edge_weight = edge_weight / (edge_weight.max() + 1e-8)
        
        return edge_index, edge_weight
    
    def get_adjacency_matrix(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> csr_matrix:
        """
        Convert edge list to sparse adjacency matrix.
        
        Args:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            num_nodes: Number of nodes
            
        Returns:
            Sparse adjacency matrix
        """
        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        data = edge_weight.cpu().numpy()
        
        return csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    def normalize_graph(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        norm_type: str = "sym",
    ) -> torch.Tensor:
        """
        Normalize edge weights.
        
        Args:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            num_nodes: Number of nodes
            norm_type: "sym" (symmetric) or "row" (row normalization)
            
        Returns:
            Normalized edge weights
        """
        row, col = edge_index
        
        # Compute degrees
        deg = torch.zeros(num_nodes, device=edge_weight.device)
        deg.scatter_add_(0, row, edge_weight)
        
        if norm_type == "sym":
            # D^{-1/2} A D^{-1/2}
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            # D^{-1} A
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            norm_weight = deg_inv[row] * edge_weight
        
        return norm_weight
