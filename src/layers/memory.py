"""
Memory-augmented neural network components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MemoryBank(nn.Module):
    """
    External memory bank for storing and retrieving representations.
    Used for global context in CONGA.
    """
    
    def __init__(
        self,
        memory_size: int,
        memory_dim: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Learnable memory slots
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        nn.init.xavier_normal_(self.memory)
        
        # Attention for reading
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        
        self.attention = nn.MultiheadAttention(
            memory_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(memory_dim)
    
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using attention.
        
        Args:
            query: [batch_size, query_dim] or [batch_size, seq_len, query_dim]
            
        Returns:
            output: [batch_size, memory_dim] or [batch_size, seq_len, memory_dim]
        """
        is_2d = query.dim() == 2
        if is_2d:
            query = query.unsqueeze(1)
        
        batch_size = query.shape[0]
        
        # Expand memory for batch
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Project query
        query_proj = self.query_proj(query)
        
        # Attend to memory
        output, _ = self.attention(query_proj, memory, memory)
        
        # Residual and norm
        output = self.layer_norm(query + output)
        
        if is_2d:
            output = output.squeeze(1)
        
        return output
    
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        learning_rate: float = 0.01,
    ):
        """
        Write to memory (update memory slots).
        
        Args:
            key: [batch_size, key_dim] keys for addressing
            value: [batch_size, value_dim] values to write
            learning_rate: rate of memory update
        """
        # Compute attention weights
        key_proj = self.key_proj(key)  # [B, D]
        
        # Similarity with memory keys
        similarity = torch.matmul(key_proj, self.memory.T)  # [B, M]
        weights = F.softmax(similarity, dim=-1)  # [B, M]
        
        # Update memory
        value_proj = self.value_proj(value)  # [B, D]
        
        # Weighted update
        update = torch.matmul(weights.T, value_proj) / (weights.sum(dim=0, keepdim=True).T + 1e-8)
        
        with torch.no_grad():
            self.memory.data = (
                (1 - learning_rate) * self.memory.data +
                learning_rate * update
            )


class MemoryAugmentedLayer(nn.Module):
    """
    Memory-augmented layer combining local and global context.
    """
    
    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 1000,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.memory_bank = MemoryBank(
            memory_size=memory_size,
            memory_dim=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        local_repr: torch.Tensor,
        update_memory: bool = False,
    ) -> torch.Tensor:
        """
        Augment local representation with global memory.
        
        Args:
            local_repr: [batch_size, hidden_size] local representation
            update_memory: whether to update memory with current representations
            
        Returns:
            augmented: [batch_size, hidden_size]
        """
        # Read from memory
        global_context = self.memory_bank.read(local_repr)
        
        # Fuse local and global
        combined = torch.cat([local_repr, global_context], dim=-1)
        fused = self.fusion(combined)
        
        # Residual
        augmented = self.layer_norm(local_repr + fused)
        
        # Optionally update memory
        if update_memory and self.training:
            self.memory_bank.write(local_repr, local_repr)
        
        return augmented
