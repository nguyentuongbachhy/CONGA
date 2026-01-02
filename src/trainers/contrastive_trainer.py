"""
Trainer for contrastive learning models (CL4SRec, GCL4SR, etc.)
"""

from typing import Dict, Any
import torch

from .base_trainer import BaseTrainer


class ContrastiveTrainer(BaseTrainer):
    """
    Trainer for models with contrastive learning objectives.
    """
    
    def __init__(
        self,
        contrastive_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.contrastive_weight = contrastive_weight
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss including contrastive component."""
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        
        # Mask for valid positions
        mask = (batch["pos_items"] != 0).float()
        
        # Recommendation loss
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        rec_loss = (
            self.criterion(pos_logits, pos_labels) * mask +
            self.criterion(neg_logits, neg_labels) * mask
        ).sum() / mask.sum()
        
        # Contrastive loss
        cl_loss = outputs.get("cl_loss", torch.tensor(0.0, device=self.device))
        
        # Total loss
        total_loss = rec_loss + self.contrastive_weight * cl_loss
        
        # Log individual losses
        if self.global_step % 100 == 0:
            self.tb_logger.log_scalar("train/rec_loss", rec_loss.item(), self.global_step)
            if isinstance(cl_loss, torch.Tensor):
                self.tb_logger.log_scalar("train/cl_loss", cl_loss.item(), self.global_step)
        
        return total_loss


class DuoTrainer(BaseTrainer):
    """
    Trainer for DuoRec model with unsupervised and supervised contrastive losses.
    """
    
    def __init__(
        self,
        unsup_weight: float = 0.1,
        sup_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.unsup_weight = unsup_weight
        self.sup_weight = sup_weight
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute DuoRec loss."""
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        
        mask = (batch["pos_items"] != 0).float()
        
        # Recommendation loss
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        rec_loss = (
            self.criterion(pos_logits, pos_labels) * mask +
            self.criterion(neg_logits, neg_labels) * mask
        ).sum() / mask.sum()
        
        # Contrastive losses
        unsup_cl_loss = outputs.get("unsup_cl_loss", torch.tensor(0.0, device=self.device))
        sup_cl_loss = outputs.get("sup_cl_loss", torch.tensor(0.0, device=self.device))
        
        # Total loss
        total_loss = (
            rec_loss +
            self.unsup_weight * unsup_cl_loss +
            self.sup_weight * sup_cl_loss
        )
        
        return total_loss


class GraphContrastiveTrainer(BaseTrainer):
    """
    Trainer for graph-based contrastive models (GCL4SR, CONGA).
    """
    
    def __init__(
        self,
        seq_cl_weight: float = 0.1,
        graph_cl_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seq_cl_weight = seq_cl_weight
        self.graph_cl_weight = graph_cl_weight
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss with sequence and graph contrastive components."""
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        
        mask = (batch["pos_items"] != 0).float()
        
        # Recommendation loss
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        rec_loss = (
            self.criterion(pos_logits, pos_labels) * mask +
            self.criterion(neg_logits, neg_labels) * mask
        ).sum() / mask.sum()
        
        # Contrastive losses
        seq_cl_loss = outputs.get("seq_cl_loss", torch.tensor(0.0, device=self.device))
        graph_cl_loss = outputs.get("graph_cl_loss", torch.tensor(0.0, device=self.device))
        
        # Total loss
        total_loss = (
            rec_loss +
            self.seq_cl_weight * seq_cl_loss +
            self.graph_cl_weight * graph_cl_loss
        )
        
        # Log
        if self.global_step % 100 == 0:
            self.tb_logger.log_scalar("train/rec_loss", rec_loss.item(), self.global_step)
            if isinstance(seq_cl_loss, torch.Tensor):
                self.tb_logger.log_scalar("train/seq_cl_loss", seq_cl_loss.item(), self.global_step)
            if isinstance(graph_cl_loss, torch.Tensor):
                self.tb_logger.log_scalar("train/graph_cl_loss", graph_cl_loss.item(), self.global_step)
        
        return total_loss
