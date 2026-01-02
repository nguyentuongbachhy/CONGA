#!/usr/bin/env python
"""
Benchmark script to run all baseline models on a dataset.

Usage:
    python scripts/benchmark.py --dataset ml-1m
    python scripts/benchmark.py --dataset beauty --models sasrec cl4srec conga
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.models import get_model
from src.data.dataset import get_dataloader
from src.trainers.base_trainer import BaseTrainer
from src.trainers.contrastive_trainer import ContrastiveTrainer, GraphContrastiveTrainer
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


# Default hyperparameters for each model
MODEL_CONFIGS = {
    "sasrec": {
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 1,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 200,
    },
    "sasrec_duo": {
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 1,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 200,
        "contrastive_weight": 0.1,
    },
    "cl4srec": {
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 1,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 200,
        "contrastive_weight": 0.1,
        "temperature": 1.0,
    },
    "gcl4sr": {
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 1,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 200,
        "num_gnn_layers": 2,
        "contrastive_weight": 0.1,
    },
    "conga": {
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 2,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 200,
        "num_local_layers": 2,
        "num_global_layers": 1,
        "contrastive_weight": 0.1,
        "graph_cl_weight": 0.1,
    },
}


def run_experiment(
    model_name: str,
    dataset: str,
    data_path: str,
    device: str,
    seed: int,
    logger,
) -> dict:
    """Run a single experiment."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {model_name} on {dataset}")
    logger.info(f"{'='*60}")
    
    set_seed(seed)
    
    # Get model config
    model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["sasrec"])
    
    # Load data
    train_loader = get_dataloader(
        data_path=data_path,
        max_seq_len=50,
        batch_size=256,
        mode="train",
    )
    
    valid_loader = get_dataloader(
        data_path=data_path,
        max_seq_len=50,
        batch_size=256,
        mode="valid",
    )
    
    test_loader = get_dataloader(
        data_path=data_path,
        max_seq_len=50,
        batch_size=256,
        mode="test",
    )
    
    num_items = train_loader.dataset.num_items
    
    # Create model
    model = get_model(
        model_name=model_name,
        num_items=num_items,
        hidden_size=model_config["hidden_size"],
        max_seq_len=50,
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout_rate=model_config["dropout_rate"],
        device=device,
    )
    
    logger.info(f"Parameters: {model.count_parameters():,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config["learning_rate"],
    )
    
    # Loss
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    
    # Create dummy config
    class DummyConfig:
        class training:
            log_dir = "experiments/logs"
            save_dir = "experiments/checkpoints"
            eval_interval = 20
            early_stopping_patience = 20
            gradient_clip = 1.0
        
        experiment_name = f"{model_name}_{dataset}"
        device = device
        use_amp = True
    
    config = DummyConfig()
    
    # Select trainer
    if model_name in ["cl4srec", "gcl4sr"]:
        trainer = ContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            contrastive_weight=model_config.get("contrastive_weight", 0.1),
        )
    elif model_name == "conga":
        trainer = GraphContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            seq_cl_weight=model_config.get("contrastive_weight", 0.1),
            graph_cl_weight=model_config.get("graph_cl_weight", 0.1),
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
        )
    
    # Train
    best_metrics = trainer.train(model_config["epochs"])
    
    return {
        "model": model_name,
        "dataset": dataset,
        "ndcg@5": best_metrics.get("ndcg@5", 0),
        "ndcg@10": best_metrics.get("ndcg@10", 0),
        "ndcg@20": best_metrics.get("ndcg@20", 0),
        "hr@5": best_metrics.get("hr@5", 0),
        "hr@10": best_metrics.get("hr@10", 0),
        "hr@20": best_metrics.get("hr@20", 0),
        "mrr@10": best_metrics.get("mrr@10", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark models")
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--models", nargs="+", 
                        default=["sasrec", "sasrec_duo", "cl4srec", "gcl4sr", "conga"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments/results/benchmark.json")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger("benchmark", "experiments/logs")
    
    data_path = f"data/{args.dataset}.txt"
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        return
    
    # Run experiments
    results = []
    
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue
        
        try:
            result = run_experiment(
                model_name=model_name,
                dataset=args.dataset,
                data_path=data_path,
                device=args.device,
                seed=args.seed,
                logger=logger,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error running {model_name}: {e}")
            continue
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*80)
    
    header = f"{'Model':<15} {'NDCG@10':<10} {'HR@10':<10} {'MRR@10':<10}"
    logger.info(header)
    logger.info("-"*50)
    
    for r in results:
        row = f"{r['model']:<15} {r['ndcg@10']:<10.4f} {r['hr@10']:<10.4f} {r['mrr@10']:<10.4f}"
        logger.info(row)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "results": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
