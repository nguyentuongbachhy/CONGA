"""
Trainer for continual learning scenarios.
"""

import copy
from typing import Dict, List, Any, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..data.continual import ContinualDataStream, ReplayBuffer, ContinualDataset
from ..losses.distillation import DistillationLoss, EWCLoss


class ContinualTrainer(BaseTrainer):
    """
    Trainer for continual learning with experience replay and distillation.
    """
    
    def __init__(
        self,
        replay_buffer_size: int = 10000,
        replay_ratio: float = 0.3,
        distillation_weight: float = 0.5,
        ewc_weight: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.replay_buffer_size = replay_buffer_size
        self.replay_ratio = replay_ratio
        self.distillation_weight = distillation_weight
        self.ewc_weight = ewc_weight
        
        # Replay buffer
        max_seq_len = self.config.data.max_seq_len if hasattr(self.config, 'data') else 50
        self.replay_buffer = ReplayBuffer(
            buffer_size=replay_buffer_size,
            max_seq_len=max_seq_len,
        )
        
        # Teacher model for distillation
        self.teacher_model = None
        
        # Distillation loss
        self.distillation_loss = DistillationLoss(
            temperature=2.0,
            alpha=distillation_weight,
            distill_type="feature",
        )
        
        # EWC loss
        if ewc_weight > 0:
            self.ewc_loss = EWCLoss(lambda_ewc=ewc_weight)
        else:
            self.ewc_loss = None
        
        # Track task performance
        self.task_metrics: List[Dict[str, float]] = []
    
    def update_teacher(self):
        """Update teacher model with current model weights."""
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.logger.info("Updated teacher model")
    
    def compute_ewc_fisher(self, loader: DataLoader):
        """Compute Fisher information for EWC."""
        if self.ewc_loss is not None:
            self.ewc_loss.compute_fisher(self.model, loader, self.device)
            self.logger.info("Computed Fisher information for EWC")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Training step with replay and distillation."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            batch["input_seq"],
            batch["pos_items"],
            batch["neg_items"],
        )
        
        # Base recommendation loss
        loss = self.compute_loss(outputs, batch)
        
        # Distillation loss
        if self.teacher_model is not None and self.distillation_weight > 0:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    batch["input_seq"],
                    batch["pos_items"],
                    batch["neg_items"],
                )
            
            distill_loss = self.distillation_loss(outputs, teacher_outputs)
            loss = loss + self.distillation_weight * distill_loss
            
            if self.global_step % 100 == 0:
                self.tb_logger.log_scalar("train/distill_loss", distill_loss.item(), self.global_step)
        
        # EWC regularization
        if self.ewc_loss is not None and self.ewc_weight > 0:
            ewc_penalty = self.ewc_loss(self.model)
            loss = loss + ewc_penalty
            
            if self.global_step % 100 == 0:
                self.tb_logger.log_scalar("train/ewc_loss", ewc_penalty.item(), self.global_step)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.config, 'training') and self.config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip
            )
        
        self.optimizer.step()
        
        # Update replay buffer
        self.update_replay_buffer(batch)
        
        return loss.item()
    
    def update_replay_buffer(self, batch: Dict[str, torch.Tensor]):
        """Add samples to replay buffer."""
        input_seq = batch["input_seq"].cpu().numpy()
        pos_items = batch["pos_items"].cpu().numpy()
        
        for i in range(len(input_seq)):
            # Use last target as sample target
            target = pos_items[i, -1] if pos_items[i, -1] != 0 else pos_items[i][pos_items[i] != 0][-1]
            self.replay_buffer.add(input_seq[i], int(target))
    
    def train_on_chunk(
        self,
        chunk_loader: DataLoader,
        num_epochs: int,
        chunk_idx: int,
    ) -> Dict[str, float]:
        """
        Train on a single data chunk.
        
        Args:
            chunk_loader: DataLoader for current chunk
            num_epochs: Epochs to train on this chunk
            chunk_idx: Index of current chunk
            
        Returns:
            Metrics on current chunk
        """
        self.logger.info(f"Training on chunk {chunk_idx}")
        
        # Update teacher before training on new chunk
        if chunk_idx > 0:
            self.update_teacher()
            
            # Compute EWC Fisher
            if self.ewc_loss is not None:
                self.compute_ewc_fisher(chunk_loader)
        
        # Train on chunk
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch_with_loader(chunk_loader)
            
            if epoch % 5 == 0:
                self.logger.info(f"Chunk {chunk_idx}, Epoch {epoch} - Loss: {train_metrics['loss']:.4f}")
        
        # Evaluate
        valid_metrics = self.evaluate(self.valid_loader, "valid")
        test_metrics = self.evaluate(self.test_loader, "test")
        
        self.logger.info(
            f"Chunk {chunk_idx} completed - "
            f"Test NDCG@10: {test_metrics['ndcg@10']:.4f}, "
            f"HR@10: {test_metrics['hr@10']:.4f}"
        )
        
        return test_metrics
    
    def train_epoch_with_loader(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with a specific loader."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            self.global_step += 1
        
        return {"loss": total_loss / num_batches}
    
    def train_continual(
        self,
        data_stream: ContinualDataStream,
        epochs_per_chunk: int = 10,
    ) -> List[Dict[str, float]]:
        """
        Train on a continual data stream.
        
        Args:
            data_stream: ContinualDataStream with temporal chunks
            epochs_per_chunk: Epochs to train on each chunk
            
        Returns:
            List of metrics for each chunk
        """
        self.logger.info(f"Starting continual training on {len(data_stream)} chunks")
        
        all_metrics = []
        
        for chunk_idx, chunk_data in enumerate(data_stream):
            # Create dataset and loader for chunk
            chunk_dataset = ContinualDataset(
                current_data=chunk_data,
                replay_buffer=self.replay_buffer if chunk_idx > 0 else None,
                replay_ratio=self.replay_ratio,
                max_seq_len=self.config.data.max_seq_len if hasattr(self.config, 'data') else 50,
                num_items=self.model.num_items,
            )
            
            chunk_loader = DataLoader(
                chunk_dataset,
                batch_size=self.config.data.train_batch_size if hasattr(self.config, 'data') else 256,
                shuffle=True,
                num_workers=4,
            )
            
            # Train on chunk
            chunk_metrics = self.train_on_chunk(
                chunk_loader,
                epochs_per_chunk,
                chunk_idx,
            )
            
            all_metrics.append(chunk_metrics)
            self.task_metrics.append(chunk_metrics)
        
        # Compute forgetting metrics
        self.compute_forgetting()
        
        return all_metrics
    
    def compute_forgetting(self):
        """Compute backward transfer (forgetting) metrics."""
        if len(self.task_metrics) < 2:
            return
        
        # Compute average forgetting
        forgetting = 0.0
        for i in range(len(self.task_metrics) - 1):
            initial = self.task_metrics[i]['ndcg@10']
            final = self.task_metrics[-1]['ndcg@10']
            forgetting += max(0, initial - final)
        
        avg_forgetting = forgetting / (len(self.task_metrics) - 1)
        
        self.logger.info(f"Average Forgetting: {avg_forgetting:.4f}")
        self.tb_logger.log_scalar("continual/forgetting", avg_forgetting, self.global_step)
