#!/usr/bin/env python
"""
Training script for CONGA and baseline models.

Usage:
    python scripts/train.py --config configs/sasrec.yaml --dataset ml-1m
    python scripts/train.py --config configs/conga.yaml --dataset beauty
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.models import get_model
from src.data.dataset import SequentialDataset, get_dataloader
from src.trainers.base_trainer import BaseTrainer
from src.trainers.contrastive_trainer import ContrastiveTrainer, GraphContrastiveTrainer
from src.trainers.continual_trainer import ContinualTrainer
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train sequential recommendation model")
    
    parser.add_argument("--config", type=str, default="configs/sasrec.yaml",
                        help="Path to config file")
    parser.add_argument("--dataset", type=str, default="ml-1m",
                        help="Dataset name")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides config)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name for logging")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with command line args
    if args.dataset:
        config.data.dataset = args.dataset
    if args.model:
        config.model.model_name = args.model
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.data.train_batch_size = args.batch_size
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        config.experiment_name = f"{config.model.model_name}_{config.data.dataset}"
    
    # Set seed
    set_seed(config.seed)
    
    # Setup logger
    logger = setup_logger(
        name="train",
        log_dir=config.training.log_dir,
    )
    
    logger.info(f"Config: {config.to_dict()}")
    
    # Data path
    data_path = os.path.join(config.data.data_dir, f"{config.data.dataset}.txt")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please download the dataset and place it in the data directory.")
        logger.info("Expected format: user_id item_id (one interaction per line)")
        return
    
    # Load data
    logger.info(f"Loading dataset: {config.data.dataset}")
    
    train_loader = get_dataloader(
        data_path=data_path,
        max_seq_len=config.data.max_seq_len,
        batch_size=config.data.train_batch_size,
        mode="train",
        num_workers=config.data.num_workers,
    )
    
    valid_loader = get_dataloader(
        data_path=data_path,
        max_seq_len=config.data.max_seq_len,
        batch_size=config.data.eval_batch_size,
        mode="valid",
        num_workers=config.data.num_workers,
    )
    
    test_loader = get_dataloader(
        data_path=data_path,
        max_seq_len=config.data.max_seq_len,
        batch_size=config.data.eval_batch_size,
        mode="test",
        num_workers=config.data.num_workers,
    )
    
    # Get dataset info
    train_dataset = train_loader.dataset
    num_items = train_dataset.num_items
    num_users = train_dataset.num_users
    
    logger.info(f"Users: {num_users}, Items: {num_items}")
    logger.info(f"Train samples: {len(train_dataset)}")
    
    # Create model
    logger.info(f"Creating model: {config.model.model_name}")
    
    model = get_model(
        model_name=config.model.model_name,
        num_items=num_items,
        hidden_size=config.model.hidden_size,
        max_seq_len=config.data.max_seq_len,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        device=config.device,
    )
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer (AdamW for better weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Scheduler
    if config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
        )
    elif config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.5,
        )
    else:
        scheduler = None
    
    # Loss
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    
    # Select trainer based on model
    model_name = config.model.model_name.lower()
    
    if model_name in ["cl4srec", "gcl4sr"]:
        trainer = ContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            scheduler=scheduler,
            contrastive_weight=config.model.contrastive_weight,
        )
    elif model_name == "conga":
        if config.model.use_continual:
            trainer = ContinualTrainer(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                config=config,
                scheduler=scheduler,
                replay_buffer_size=config.model.memory_size,
                replay_ratio=config.model.replay_ratio,
                distillation_weight=config.model.distillation_weight,
            )
        else:
            trainer = GraphContrastiveTrainer(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                config=config,
                scheduler=scheduler,
                seq_cl_weight=config.model.contrastive_weight,
                graph_cl_weight=config.model.contrastive_weight,
            )
    else:
        trainer = BaseTrainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            scheduler=scheduler,
        )
    
    # Train
    logger.info("Starting training...")
    best_metrics = trainer.train(config.training.epochs)
    
    # Final results
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Best Test NDCG@10: {best_metrics.get('ndcg@10', 0):.4f}")
    logger.info(f"Best Test HR@10: {best_metrics.get('hr@10', 0):.4f}")
    logger.info("=" * 50)
    
    # Save final config
    config_save_path = os.path.join(config.training.save_dir, "config.yaml")
    config.to_yaml(config_save_path)


if __name__ == "__main__":
    main()
