"""
Early stopping utility.
"""

import os
import torch
import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        save_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "max" for metrics like accuracy, "min" for loss
            save_path: Path to save best model
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == "max":
            self.is_better = lambda new, old: new > old + min_delta
            self.best_score = float('-inf')
        else:
            self.is_better = lambda new, old: new < old - min_delta
            self.best_score = float('inf')
    
    def __call__(
        self,
        score: float,
        model: torch.nn.Module,
        epoch: int,
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save
            epoch: Current epoch
            
        Returns:
            True if should stop, False otherwise
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
            if self.save_path:
                self.save_checkpoint(model, epoch, score)
            
            if self.verbose:
                print(f"EarlyStopping: New best score {score:.4f} at epoch {epoch}")
            
            return False
        else:
            self.counter += 1
            
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping! Best score {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
            
            return False
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        score: float,
    ):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Ensure pure Python scalars to avoid torch.load(weights_only=True) issues
        # with numpy scalar types in newer PyTorch versions.
        epoch = int(epoch)
        score = float(score)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_score": score,
        }
        
        torch.save(checkpoint, self.save_path)
        
        if self.verbose:
            print(f"Saved checkpoint to {self.save_path}")
    
    def load_best(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load best model checkpoint."""
        if self.save_path and os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            if self.verbose:
                print(f"Loaded best model from epoch {checkpoint['epoch']} "
                      f"with score {checkpoint['best_score']:.4f}")
        
        return model
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
        if self.mode == "max":
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')
