"""
Base trainer for sequential recommendation models.
"""

import os
import time
from typing import Dict, Optional, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from ..utils.metrics import MetricTracker
from ..utils.early_stopping import EarlyStopping
from ..utils.logger import setup_logger, TensorBoardLogger
from ..utils.seed import set_seed


class BaseTrainer:
    """
    Base trainer class for all models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Any,
        scheduler: Optional[Any] = None,
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer
            criterion: Loss function
            config: Configuration object
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.scheduler = scheduler
        
        # Device
        self.device = config.device if hasattr(config, 'device') else 'cuda'
        self.model = self.model.to(self.device)
        
        # AMP
        self.use_amp = config.use_amp if hasattr(config, 'use_amp') else True
        # AMP only makes sense on CUDA
        if str(self.device).startswith("cpu"):
            self.use_amp = False
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.logger = setup_logger(
            name="trainer",
            log_dir=config.training.log_dir if hasattr(config, 'training') else "experiments/logs",
        )
        
        self.tb_logger = TensorBoardLogger(
            log_dir=config.training.log_dir if hasattr(config, 'training') else "experiments/logs",
            experiment_name=config.experiment_name if hasattr(config, 'experiment_name') else "default",
        )
        
        # Early stopping
        save_path = os.path.join(
            config.training.save_dir if hasattr(config, 'training') else "experiments/checkpoints",
            "best_model.pt"
        )
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience if hasattr(config, 'training') else 20,
            mode="max",
            save_path=save_path,
        )
        
        # Metrics
        self.metric_tracker = MetricTracker(ks=[5, 10, 20])
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            loss = self.train_step(batch)
            
            total_loss += loss
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            # Log to tensorboard
            self.tb_logger.log_scalar("train/loss", loss, self.global_step)
        
        avg_loss = total_loss / num_batches
        
        return {"loss": avg_loss}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with autocast():
                outputs = self.model(
                    batch["input_seq"],
                    batch["pos_items"],
                    batch["neg_items"],
                )
                loss = self.compute_loss(outputs, batch)
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if hasattr(self.config, 'training') and self.config.training.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                batch["input_seq"],
                batch["pos_items"],
                batch["neg_items"],
            )
            loss = self.compute_loss(outputs, batch)
            
            loss.backward()
            
            if hasattr(self.config, 'training') and self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
        
        return loss.item()
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss from model outputs."""
        pos_logits = outputs["pos_logits"]
        neg_logits = outputs["neg_logits"]
        
        # Mask for valid positions
        mask = (batch["pos_items"] != 0).float()
        
        # BCE loss
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        
        loss = (
            self.criterion(pos_logits, pos_labels) * mask +
            self.criterion(neg_logits, neg_labels) * mask
        ).sum() / mask.sum()
        
        return loss
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "valid") -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        self.metric_tracker.reset()
        
        for batch in tqdm(loader, desc=f"Evaluating {split}"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get predictions
            scores = self.model.predict(batch["input_seq"])
            
            # Get target scores and negative scores
            target_item = batch["target_item"]
            batch_size = scores.shape[0]
            
            # Create candidate list: [target, 100 random negatives]
            target_scores = scores.gather(1, target_item.unsqueeze(1))

            # Negatives: prefer dataset-provided negatives (guaranteed not in user history)
            if "neg_items" in batch:
                neg_indices = batch["neg_items"]
                if neg_indices.dim() == 1:
                    neg_indices = neg_indices.unsqueeze(0)
                neg_scores = scores.gather(1, neg_indices)
            else:
                # Fallback: sample random negatives (may include positives)
                num_neg = 100
                # Avoid sampling padding index 0
                neg_indices = torch.randint(1, scores.shape[1], (batch_size, num_neg), device=self.device)
                neg_scores = scores.gather(1, neg_indices)
            
            # Combine
            combined_scores = torch.cat([target_scores, neg_scores], dim=1)
            
            # Update metrics (target at index 0)
            self.metric_tracker.update_batch(combined_scores)
        
        metrics = self.metric_tracker.compute()
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Best metrics achieved
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Evaluate
            eval_interval = self.config.training.eval_interval if hasattr(self.config, 'training') else 1
            if epoch % eval_interval == 0:
                valid_metrics = self.evaluate(self.valid_loader, "valid")
                test_metrics = self.evaluate(self.test_loader, "test")
                
                # Log metrics
                self.logger.info(
                    f"Epoch {epoch} - "
                    f"Valid NDCG@10: {valid_metrics['ndcg@10']:.4f}, "
                    f"Valid HR@10: {valid_metrics['hr@10']:.4f} | "
                    f"Test NDCG@10: {test_metrics['ndcg@10']:.4f}, "
                    f"Test HR@10: {test_metrics['hr@10']:.4f}"
                )
                
                self.tb_logger.log_metrics(valid_metrics, "valid", self.global_step)
                self.tb_logger.log_metrics(test_metrics, "test", self.global_step)
                
                # Early stopping
                if self.early_stopping(valid_metrics['ndcg@10'], self.model, epoch):
                    self.logger.info("Early stopping triggered")
                    break
                
                # Update best metrics
                if valid_metrics['ndcg@10'] > self.best_metrics.get('valid_ndcg@10', 0):
                    self.best_metrics = {
                        'valid_ndcg@10': valid_metrics['ndcg@10'],
                        'valid_hr@10': valid_metrics['hr@10'],
                        'test_ndcg@10': test_metrics['ndcg@10'],
                        'test_hr@10': test_metrics['hr@10'],
                        'epoch': epoch,
                    }
        
        # Load best model
        self.early_stopping.load_best(self.model)
        
        # Final evaluation
        final_test_metrics = self.evaluate(self.test_loader, "test")
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        self.logger.info(f"Best Test NDCG@10: {final_test_metrics['ndcg@10']:.4f}")
        
        self.tb_logger.close()
        
        return final_test_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metrics': self.best_metrics,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metrics = checkpoint.get('best_metrics', {})
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
