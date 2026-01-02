"""
Knowledge distillation loss for continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for continual learning.
    
    Transfers knowledge from teacher (previous model) to student (current model).
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        distill_type: str = "soft",
    ):
        """
        Args:
            temperature: Softmax temperature for soft distillation
            alpha: Weight for distillation loss vs task loss
            distill_type: "soft" (logit matching), "feature" (representation matching),
                         or "attention" (attention map matching)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.distill_type = distill_type
    
    def soft_distillation(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Soft target distillation (Hinton et al.).
        
        Args:
            student_logits: [batch_size, num_classes]
            teacher_logits: [batch_size, num_classes]
            
        Returns:
            loss: scalar
        """
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        loss = loss * (self.temperature ** 2)
        
        return loss
    
    def feature_distillation(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Feature/representation distillation.
        
        Args:
            student_features: [batch_size, hidden_size]
            teacher_features: [batch_size, hidden_size]
            normalize: Whether to normalize features
            
        Returns:
            loss: scalar
        """
        if normalize:
            student_features = F.normalize(student_features, dim=-1)
            teacher_features = F.normalize(teacher_features, dim=-1)
        
        loss = F.mse_loss(student_features, teacher_features)
        
        return loss
    
    def attention_distillation(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention map distillation.
        
        Args:
            student_attention: [batch_size, num_heads, seq_len, seq_len]
            teacher_attention: [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            loss: scalar
        """
        # Flatten attention maps
        student_flat = student_attention.view(student_attention.size(0), -1)
        teacher_flat = teacher_attention.view(teacher_attention.size(0), -1)
        
        # Normalize
        student_flat = F.softmax(student_flat, dim=-1)
        teacher_flat = F.softmax(teacher_flat, dim=-1)
        
        loss = F.kl_div(student_flat.log(), teacher_flat, reduction='batchmean')
        
        return loss
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            
        Returns:
            loss: scalar
        """
        if self.distill_type == "soft":
            # Use logits
            student_logits = student_outputs.get("pos_logits")
            teacher_logits = teacher_outputs.get("pos_logits")
            
            if student_logits is None or teacher_logits is None:
                return torch.tensor(0.0, device=next(iter(student_outputs.values())).device)
            
            return self.soft_distillation(student_logits, teacher_logits)
        
        elif self.distill_type == "feature":
            # Use sequence representations
            student_feat = student_outputs.get("seq_repr", student_outputs.get("final_repr"))
            teacher_feat = teacher_outputs.get("seq_repr", teacher_outputs.get("final_repr"))
            
            if student_feat is None or teacher_feat is None:
                return torch.tensor(0.0, device=next(iter(student_outputs.values())).device)
            
            return self.feature_distillation(student_feat, teacher_feat)
        
        elif self.distill_type == "attention":
            student_attn = student_outputs.get("attention_weights")
            teacher_attn = teacher_outputs.get("attention_weights")
            
            if student_attn is None or teacher_attn is None:
                return torch.tensor(0.0, device=next(iter(student_outputs.values())).device)
            
            return self.attention_distillation(student_attn, teacher_attn)
        
        else:
            raise ValueError(f"Unknown distillation type: {self.distill_type}")


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation (EWC) loss for continual learning.
    
    Penalizes changes to important parameters.
    """
    
    def __init__(
        self,
        lambda_ewc: float = 1000.0,
    ):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        
        # Storage for fisher information and old parameters
        self.fisher = {}
        self.old_params = {}
    
    def compute_fisher(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda",
    ) -> None:
        """
        Compute Fisher information matrix.
        
        Args:
            model: The model
            dataloader: Data loader for computing Fisher
            device: Device to use
        """
        self.fisher = {}
        self.old_params = {}
        
        model.eval()
        
        # Initialize Fisher
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
                self.old_params[name] = param.clone().detach()
        
        # Compute Fisher
        for batch in dataloader:
            model.zero_grad()
            
            # Forward pass
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            # Use log-likelihood as objective
            loss = outputs.get("loss", outputs.get("pos_logits", torch.tensor(0.0)).mean())
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.pow(2)
        
        # Normalize
        num_batches = len(dataloader)
        for name in self.fisher:
            self.fisher[name] /= num_batches
        
        model.train()
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty.
        
        Args:
            model: Current model
            
        Returns:
            penalty: scalar
        """
        if not self.fisher:
            return torch.tensor(0.0)
        
        penalty = 0.0
        
        for name, param in model.named_parameters():
            if name in self.fisher:
                penalty += (
                    self.fisher[name] * 
                    (param - self.old_params[name]).pow(2)
                ).sum()
        
        return self.lambda_ewc * penalty
