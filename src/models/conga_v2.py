"""
CONGA v2: Enhanced version with continual learning improvements.

Improvements over v1:
1. Lazy data loading support (reduced RAM usage)
2. PReLU activation (better gradient flow)
3. AdamW optimizer (better weight decay)
4. Memory consolidation with importance weighting
5. Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention
6. Nested learning with expert networks
7. Structured checkpoint saving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .conga import (
    CONGA,
    LocalItemGraph,
    GlobalUserItemGraph,
    NestedGraphEncoder,
    MultiScaleContrastive,
)
from .memory_consolidation import (
    ImportanceWeightedMemory,
    ElasticWeightConsolidation,
    NestedLearningModule,
)


class CONGAv2(CONGA):
    """
    Enhanced CONGA with continual learning capabilities.
    
    Key additions:
    - Importance-weighted memory replay
    - EWC for parameter protection
    - Nested learning with expert networks
    - Better activation functions (PReLU)
    """
    
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 64,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        device: str = "cuda",
        # Graph parameters
        num_local_layers: int = 2,
        num_global_layers: int = 1,
        memory_bank_size: int = 1000,
        # Contrastive parameters
        contrastive_weight: float = 0.1,
        graph_cl_weight: float = 0.1,
        temperature: float = 0.07,
        # Continual learning parameters (enhanced)
        use_continual: bool = True,
        replay_buffer_size: int = 10000,
        replay_ratio: float = 0.3,
        distillation_weight: float = 0.5,
        # New v2 parameters
        use_ewc: bool = True,
        ewc_lambda: float = 1000.0,
        use_nested_learning: bool = True,
        num_experts: int = 4,
        importance_alpha: float = 0.6,
    ):
        # Initialize base CONGA (already has PReLU from previous edits)
        super().__init__(
            num_items=num_items,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            device=device,
            num_local_layers=num_local_layers,
            num_global_layers=num_global_layers,
            memory_bank_size=memory_bank_size,
            contrastive_weight=contrastive_weight,
            graph_cl_weight=graph_cl_weight,
            temperature=temperature,
            use_continual=False,  # We'll use enhanced version
            replay_buffer_size=replay_buffer_size,
            replay_ratio=replay_ratio,
            distillation_weight=distillation_weight,
        )
        
        # CONGA (v1) continual-learning code path expects `self.teacher` to exist.
        # CONGAv2 uses its own continual-learning components, so keep teacher as None.
        self.teacher = None

        # v2 continual learning switches
        self.use_continual = use_continual
        self.use_ewc = use_ewc
        self.use_nested_learning = use_nested_learning
        
        # Enhanced continual learning components
        if use_continual:
            # Importance-weighted memory
            self.memory_replay = ImportanceWeightedMemory(
                buffer_size=replay_buffer_size,
                max_seq_len=max_seq_len,
                hidden_size=hidden_size,
                alpha=importance_alpha,
            )
            
            # EWC for parameter protection
            if use_ewc:
                self.ewc = ElasticWeightConsolidation(
                    model=self,
                    lambda_ewc=ewc_lambda,
                )
            
            # Nested learning module
            if use_nested_learning:
                self.nested_learning = NestedLearningModule(
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    dropout_rate=dropout_rate,
                )
    
    def forward(
        self,
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with continual learning.
        """
        # Base CONGA forward (disable CONGA v1 continual-learning branch)
        _v2_use_continual = self.use_continual
        self.use_continual = False
        try:
            outputs = super().forward(item_seq, pos_items, neg_items)
        finally:
            self.use_continual = _v2_use_continual
        
        # Apply nested learning if enabled
        if self.use_continual and self.use_nested_learning:
            final_repr = outputs["final_repr"]
            enhanced_repr = self.nested_learning(final_repr)
            
            # Recompute logits with enhanced representation
            pos_emb = self.item_embedding(pos_items[:, -1:])
            neg_emb = self.item_embedding(neg_items[:, -1:])
            
            pos_logits_enhanced = (enhanced_repr.unsqueeze(1) * pos_emb).sum(dim=-1)
            neg_logits_enhanced = (enhanced_repr.unsqueeze(1) * neg_emb).sum(dim=-1)
            
            outputs["pos_logits_nested"] = pos_logits_enhanced
            outputs["neg_logits_nested"] = neg_logits_enhanced
            outputs["enhanced_repr"] = enhanced_repr
        
        return outputs
    
    def compute_total_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        pos_items: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Enhanced loss computation with EWC penalty.
        """
        # Base CONGA loss (disable CONGA v1 continual-learning branch)
        _v2_use_continual = self.use_continual
        self.use_continual = False
        try:
            total_loss = super().compute_total_loss(outputs, pos_items, criterion)
        finally:
            self.use_continual = _v2_use_continual
        
        # Add EWC penalty if enabled
        if self.use_continual and self.use_ewc and hasattr(self, 'ewc'):
            ewc_penalty = self.ewc.penalty()
            total_loss += ewc_penalty
            outputs["ewc_loss"] = ewc_penalty
        
        # Add nested learning loss if available
        if "pos_logits_nested" in outputs:
            mask = (pos_items[:, -1:] != 0).float()
            
            pos_labels = torch.ones_like(outputs["pos_logits_nested"])
            neg_labels = torch.zeros_like(outputs["neg_logits_nested"])
            
            nested_loss = (
                criterion(outputs["pos_logits_nested"], pos_labels) * mask +
                criterion(outputs["neg_logits_nested"], neg_labels) * mask
            ).sum() / mask.sum()
            
            total_loss += 0.1 * nested_loss  # Weight for nested learning
            outputs["nested_loss"] = nested_loss
        
        return total_loss
    
    def update_memory(
        self,
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        seq_repr: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Update importance-weighted memory with current batch.
        
        Args:
            item_seq: [batch_size, seq_len]
            pos_items: [batch_size, seq_len]
            seq_repr: [batch_size, hidden_size]
            loss: [batch_size] individual sample losses
        """
        if self.use_continual:
            self.memory_replay.add(item_seq, pos_items, seq_repr, loss)
    
    def get_replay_batch(
        self,
        batch_size: int,
        use_diversity: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get replay samples with importance weighting.
        
        Args:
            batch_size: Number of samples to retrieve
            use_diversity: Whether to use diversity-based sampling
            
        Returns:
            seq_batch, pos_batch, weights (None if diversity sampling)
        """
        if not self.use_continual:
            return None, None, None
        
        if use_diversity:
            seq_batch, pos_batch = self.memory_replay.get_diverse_sample(batch_size)
            return seq_batch, pos_batch, None
        else:
            return self.memory_replay.sample(batch_size)
    
    def consolidate_knowledge(self, dataloader, device: str = "cpu"):
        """
        Consolidate knowledge from current task using EWC.
        
        Should be called after training on a task/time period.
        """
        if self.use_continual and self.use_ewc:
            self.ewc.compute_fisher(dataloader, device)
    
    def add_expert(self):
        """
        Add a new expert for nested learning.
        
        Useful when moving to a new time period or user segment.
        """
        if self.use_continual and self.use_nested_learning:
            self.nested_learning.add_expert()
