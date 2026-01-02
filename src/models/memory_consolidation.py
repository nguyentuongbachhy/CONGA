"""
Memory Consolidation Module for Continual Learning in CONGA.

Prevents catastrophic forgetting by:
1. Structured memory bank with importance weighting
2. Elastic Weight Consolidation (EWC) for parameter protection
3. Progressive neural networks approach for nested learning
4. Hindsight-inspired experience replay with goal relabeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque


class ImportanceWeightedMemory(nn.Module):
    """
    Structured memory bank with importance weighting.
    
    Instead of random sampling, prioritizes important experiences based on:
    - Prediction error (hard examples)
    - Temporal diversity (spread across time)
    - Representation diversity (diverse user behaviors)
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        max_seq_len: int = 50,
        hidden_size: int = 64,
        alpha: float = 0.6,  # Importance sampling exponent
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.alpha = alpha
        
        # Structured buffers
        self.register_buffer("seq_buffer", torch.zeros(buffer_size, max_seq_len, dtype=torch.long))
        self.register_buffer("pos_buffer", torch.zeros(buffer_size, max_seq_len, dtype=torch.long))
        self.register_buffer("repr_buffer", torch.zeros(buffer_size, hidden_size))
        self.register_buffer("importance", torch.zeros(buffer_size))
        self.register_buffer("timestamps", torch.zeros(buffer_size, dtype=torch.long))
        self.register_buffer("buffer_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("buffer_count", torch.zeros(1, dtype=torch.long))
        
        self.current_time = 0
    
    def add(
        self,
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        seq_repr: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Add samples to structured memory with importance weighting.
        
        Args:
            item_seq: [batch_size, seq_len]
            pos_items: [batch_size, seq_len]
            seq_repr: [batch_size, hidden_size] sequence representations
            loss: [batch_size] individual sample losses (importance indicator)
        """
        batch_size = item_seq.shape[0]
        
        for i in range(batch_size):
            ptr = int(self.buffer_ptr.item())
            
            # Store experience
            self.seq_buffer[ptr] = item_seq[i]
            self.pos_buffer[ptr] = pos_items[i]
            self.repr_buffer[ptr] = seq_repr[i].detach()
            
            # Importance = loss (higher loss = more important)
            self.importance[ptr] = loss[i].item()
            self.timestamps[ptr] = self.current_time
            
            self.buffer_ptr[0] = (ptr + 1) % self.buffer_size
            self.buffer_count[0] = min(self.buffer_count[0] + 1, self.buffer_size)
        
        self.current_time += 1
    
    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,  # Importance sampling bias correction
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from buffer using importance-based prioritization.
        
        Returns:
            seq_batch: [batch_size, seq_len]
            pos_batch: [batch_size, seq_len]
            weights: [batch_size] importance sampling weights
        """
        count = int(self.buffer_count.item())
        if count == 0:
            return None, None, None
        
        # Compute sampling probabilities
        priorities = self.importance[:count] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Importance sampling weights for bias correction
        weights = (count * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        return (
            self.seq_buffer[indices],
            self.pos_buffer[indices],
            weights,
        )
    
    def get_diverse_sample(
        self,
        batch_size: int,
        diversity_weight: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample diverse experiences based on representation diversity.
        
        Uses k-means clustering in representation space to ensure diversity.
        """
        count = int(self.buffer_count.item())
        if count == 0 or count < batch_size:
            return self.sample(batch_size)[:2]
        
        # Simple diversity sampling: maximize distance in representation space
        selected_indices = []
        available = torch.arange(count)
        
        # Start with highest importance sample
        first_idx = self.importance[:count].argmax()
        selected_indices.append(first_idx)
        
        for _ in range(batch_size - 1):
            if len(available) == 0:
                break
            
            # Compute distance to already selected samples
            selected_reprs = self.repr_buffer[selected_indices]
            available_reprs = self.repr_buffer[available]
            
            # Distance matrix: [available, selected]
            dists = torch.cdist(available_reprs.unsqueeze(0), selected_reprs.unsqueeze(0)).squeeze(0)
            min_dists = dists.min(dim=1)[0]
            
            # Combine importance and diversity
            importance_scores = self.importance[available]
            combined_scores = (1 - diversity_weight) * importance_scores + diversity_weight * min_dists
            
            # Select next sample
            next_idx = combined_scores.argmax()
            selected_indices.append(available[next_idx].item())
            available = available[available != available[next_idx]]
        
        selected_indices = torch.tensor(selected_indices, dtype=torch.long)
        
        return (
            self.seq_buffer[selected_indices],
            self.pos_buffer[selected_indices],
        )


class ElasticWeightConsolidation(nn.Module):
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.
    
    Protects important parameters learned from previous tasks by adding
    a quadratic penalty based on Fisher Information Matrix.
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        super().__init__()
        # IMPORTANT: don't register the full model as a submodule, otherwise
        # model.to(...) will recurse indefinitely (model -> ewc -> model -> ...).
        object.__setattr__(self, "_model", model)
        self.lambda_ewc = lambda_ewc
        
        # Store important parameters
        self.saved_params = {}
        self.fisher_matrix = {}
        
    def compute_fisher(
        self,
        dataloader,
        device: str = "cpu",
        num_samples: int = 1000,
    ):
        """
        Compute Fisher Information Matrix for current task.
        
        Approximates Fisher as the gradient of log-likelihood squared.
        """
        self._model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self._model.named_parameters() if p.requires_grad}
        
        samples_seen = 0
        for batch in dataloader:
            if samples_seen >= num_samples:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self._model(
                batch["input_seq"],
                batch["pos_items"],
                batch["neg_items"],
            )
            
            # Compute loss
            loss = outputs["pos_logits"].sum()
            
            # Backward to get gradients
            self._model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher approximation)
            for n, p in self._model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            
            samples_seen += batch["input_seq"].shape[0]
        
        # Normalize by number of samples
        for n in fisher:
            fisher[n] /= samples_seen
        
        self.fisher_matrix = fisher
        self.saved_params = {n: p.clone().detach() for n, p in self._model.named_parameters() if p.requires_grad}
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty: sum of Fisher-weighted squared parameter changes.
        """
        if not self.fisher_matrix:
            return torch.tensor(0.0)
        
        loss = 0.0
        for n, p in self._model.named_parameters():
            if n in self.fisher_matrix:
                loss += (self.fisher_matrix[n] * (p - self.saved_params[n]) ** 2).sum()
        
        return self.lambda_ewc * loss


class NestedLearningModule(nn.Module):
    """
    Nested/Progressive learning approach for continual learning.
    
    Maintains multiple "expert" sub-networks that specialize in different
    time periods or user segments, with a gating mechanism to combine them.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Expert networks (lightweight adapters)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.PReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1),
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert mixing.
        
        Args:
            x: [batch_size, hidden_size]
            
        Returns:
            output: [batch_size, hidden_size]
        """
        # Compute gating weights
        gate_weights = self.gate(x)  # [B, num_experts]
        
        # Apply each expert
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # Stack and weight
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, H]
        
        # Weighted combination
        output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
        
        # Residual connection
        output = self.layer_norm(output + x)
        
        return output
    
    def add_expert(self):
        """Add a new expert for new task/time period."""
        new_expert = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.experts.append(new_expert)
        self.num_experts += 1
        
        # Update gate
        old_gate_weight = self.gate[0].weight.data
        old_gate_bias = self.gate[0].bias.data
        
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_experts),
            nn.Softmax(dim=-1),
        )
        
        # Initialize new gate with old weights
        with torch.no_grad():
            self.gate[0].weight.data[:self.num_experts-1] = old_gate_weight
            self.gate[0].bias.data[:self.num_experts-1] = old_gate_bias
