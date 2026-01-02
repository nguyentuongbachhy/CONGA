"""
Contrastive learning components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    Maps representations to a lower-dimensional space for contrastive loss.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
            
        Returns:
            z: [batch_size, output_dim]
        """
        return self.projector(x)


class ContrastiveHead(nn.Module):
    """
    Contrastive learning head with projection and loss computation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        self.projector = ProjectionHead(input_dim, hidden_dim, output_dim)
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        return_loss: bool = True,
    ):
        """
        Args:
            z1: [batch_size, input_dim] first view
            z2: [batch_size, input_dim] second view
            return_loss: whether to compute loss
            
        Returns:
            If return_loss:
                loss: scalar
            Else:
                (proj_z1, proj_z2): projected representations
        """
        # Project
        proj_z1 = self.projector(z1)
        proj_z2 = self.projector(z2)
        
        if not return_loss:
            return proj_z1, proj_z2
        
        # Normalize
        proj_z1 = F.normalize(proj_z1, dim=-1)
        proj_z2 = F.normalize(proj_z2, dim=-1)
        
        batch_size = proj_z1.shape[0]
        
        # Compute similarity
        sim_matrix = torch.matmul(proj_z1, proj_z2.T) / self.temperature
        
        # InfoNCE loss
        labels = torch.arange(batch_size, device=proj_z1.device)
        loss = (
            F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.T, labels)
        ) / 2
        
        return loss


class MomentumEncoder(nn.Module):
    """
    Momentum encoder for MoCo-style contrastive learning.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        momentum: float = 0.999,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.momentum_encoder = self._copy_encoder(encoder)
        self.momentum = momentum
        
        # Freeze momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
    
    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        """Create a copy of the encoder."""
        import copy
        return copy.deepcopy(encoder)
    
    @torch.no_grad()
    def update_momentum_encoder(self):
        """Update momentum encoder with EMA."""
        for param, mom_param in zip(
            self.encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            mom_param.data = (
                self.momentum * mom_param.data +
                (1 - self.momentum) * param.data
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through main encoder."""
        return self.encoder(x)
    
    @torch.no_grad()
    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through momentum encoder."""
        return self.momentum_encoder(x)
