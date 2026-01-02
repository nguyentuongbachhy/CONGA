"""
Logging utilities.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import contextlib
import io

import torch


def setup_logger(
    name: str = "conga",
    log_dir: str = "experiments/logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TensorBoardLogger:
    """
    TensorBoard logging wrapper.
    """
    
    def __init__(
        self,
        log_dir: str = "experiments/logs/tensorboard",
        experiment_name: str = "default",
    ):
        """
        Args:
            log_dir: Base log directory
            experiment_name: Name of experiment
        """
        self.writer = None
        self.step = 0

        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                from torch.utils.tensorboard import SummaryWriter
        except Exception:
            self.log_path = None
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.log_path, exist_ok=True)

        self.writer = SummaryWriter(self.log_path)
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
    ):
        """Log a scalar value."""
        if self.writer is None:
            return
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log multiple scalars."""
        if self.writer is None:
            return
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = "eval",
        step: Optional[int] = None,
    ):
        """Log evaluation metrics."""
        if self.writer is None:
            return
        if step is None:
            step = self.step
        
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: Optional[int] = None,
    ):
        """Log histogram of values."""
        if self.writer is None:
            return
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_embedding(
        self,
        tag: str,
        embeddings: torch.Tensor,
        labels: Optional[list] = None,
        step: Optional[int] = None,
    ):
        """Log embeddings for visualization."""
        if self.writer is None:
            return
        if step is None:
            step = self.step
        self.writer.add_embedding(embeddings, metadata=labels, tag=tag, global_step=step)
    
    def log_hparams(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float],
    ):
        """Log hyperparameters with final metrics."""
        if self.writer is None:
            return
        self.writer.add_hparams(hparams, metrics)
    
    def increment_step(self):
        """Increment global step."""
        self.step += 1
    
    def set_step(self, step: int):
        """Set global step."""
        self.step = step
    
    def close(self):
        """Close the writer."""
        if self.writer is None:
            return
        self.writer.close()


class WandbLogger:
    """
    Weights & Biases logging wrapper.
    """
    
    def __init__(
        self,
        project: str = "conga",
        experiment_name: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            project: W&B project name
            experiment_name: Run name
            config: Configuration dictionary
        """
        import wandb
        
        self.run = wandb.init(
            project=project,
            name=experiment_name,
            config=config,
        )
        self.step = 0
    
    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
    ):
        """Log data."""
        import wandb
        
        if step is None:
            step = self.step
        
        wandb.log(data, step=step)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = "eval",
        step: Optional[int] = None,
    ):
        """Log metrics with prefix."""
        data = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.log(data, step)
    
    def increment_step(self):
        """Increment step."""
        self.step += 1
    
    def finish(self):
        """Finish run."""
        import wandb
        wandb.finish()
