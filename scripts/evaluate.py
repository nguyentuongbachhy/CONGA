#!/usr/bin/env python
"""
Evaluation script for trained models.

Usage:
    python scripts/evaluate.py --checkpoint experiments/checkpoints/best_model.pt --dataset ml-1m
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.models import get_model
from src.data.dataset import get_dataloader
from src.utils.metrics import MetricTracker, compute_metrics
from src.utils.seed import set_seed
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name")
    parser.add_argument("--model", type=str, default="sasrec",
                        help="Model architecture")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_negatives", type=int, default=100,
                        help="Number of negative samples for evaluation")
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20],
                        help="Cutoffs for metrics")
    
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model,
    test_loader,
    device: str,
    num_negatives: int = 100,
    ks: list = [5, 10, 20],
):
    """Evaluate model on test set."""
    model.eval()
    tracker = MetricTracker(ks=ks)
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_seq = batch["input_seq"].to(device)
        target_item = batch["target_item"].to(device)
        batch_size = input_seq.shape[0]
        
        # Get predictions for all items
        scores = model.predict(input_seq)
        
        # Get target scores
        target_scores = scores.gather(1, target_item.unsqueeze(1))

        if "neg_items" in batch:
            neg_indices = batch["neg_items"].to(device)
            if neg_indices.dim() == 1:
                neg_indices = neg_indices.unsqueeze(0)
            neg_scores = scores.gather(1, neg_indices)
        else:
            neg_indices = torch.randint(
                1, scores.shape[1],
                (batch_size, num_negatives),
                device=device
            )
            neg_scores = scores.gather(1, neg_indices)
        
        # Combine: [target, negatives]
        combined_scores = torch.cat([target_scores, neg_scores], dim=1)
        
        # Update metrics
        tracker.update_batch(combined_scores)
    
    return tracker.compute()


def main():
    args = parse_args()
    set_seed(42)
    
    logger = setup_logger("evaluate", "experiments/logs", file=False)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
    
    # PyTorch >= 2.6 defaults torch.load(weights_only=True), which can fail for
    # legacy checkpoints containing non-tensor objects (e.g., numpy scalars).
    checkpoint = torch.load(
        args.checkpoint,
        map_location=args.device,
        weights_only=False,
    )
    
    # Load data
    data_path = f"data/{args.dataset}.txt"
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        return
    
    test_loader = get_dataloader(
        data_path=data_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        mode="test",
    )
    
    num_items = test_loader.dataset.num_items
    
    # Create model
    model = get_model(
        model_name=args.model,
        num_items=num_items,
        hidden_size=args.hidden_size,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        device=args.device,
    )
    
    # Load weights
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    
    logger.info(f"Loaded model from {args.checkpoint}")
    if isinstance(checkpoint, dict):
        logger.info(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    metrics = evaluate(
        model,
        test_loader,
        args.device,
        args.num_negatives,
        args.ks,
    )
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    
    for k in args.ks:
        logger.info(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        logger.info(f"HR@{k}: {metrics[f'hr@{k}']:.4f}")
        logger.info(f"MRR@{k}: {metrics[f'mrr@{k}']:.4f}")
        logger.info("-"*30)


if __name__ == "__main__":
    main()
