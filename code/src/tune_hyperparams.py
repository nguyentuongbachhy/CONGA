"""
Hyperparameter tuning script for pattern-based training
Stage 1: Tune pattern_init_alpha (100 epochs each)
Stage 2: Tune pattern_reg_weight with best alpha (100 epochs each)

All experiments use fixed seed=42 for fair comparison.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from model import SASRec
from pattern_utils import PatternAwareInitializer, PatternRegularizer
from sasrec_integration import initialize_sasrec_with_graph_embeddings
from utils import (
    check_and_convert_dataset,
    load_metadata,
    get_dataloader,
    data_partition,
    evaluate,
    evaluate_valid,
)


def str2bool(s: str) -> bool:
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gbce_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
              alpha=0.75, temperature=1.0, eps=1e-10) -> torch.Tensor:
    """Generalized Binary Cross-Entropy loss"""
    y_pred_scaled = y_pred / temperature
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_loss = -(y_true * torch.log(sigmoid_pred + eps) +
                 (1 - y_true) * torch.log(1 - sigmoid_pred + eps))
    bce_loss = torch.mean(bce_loss)
    ce_loss = -F.log_softmax(y_pred_scaled, dim=1)[:, 0]
    ce_loss = torch.mean(ce_loss)
    total_loss = alpha * bce_loss + (1 - alpha) * ce_loss
    return total_loss


def train_with_config(args, alpha, reg_weight, stage_name):
    """
    Train one configuration and return validation NDCG
    
    Args:
        args: Base arguments
        alpha: pattern_init_alpha value
        reg_weight: pattern_reg_weight value
        stage_name: Name for this experiment
        
    Returns:
        dict with results
    """
    print(f"\n{'='*80}")
    print(f"TUNING: {stage_name}")
    print(f"{'='*80}")
    print(f"Alpha: {alpha} | Reg Weight: {reg_weight} | Epochs: {args.num_epochs}")
    print(f"Seed: {args.seed} (fixed)")
    print(f"{'='*80}\n")
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    
    train_loader = get_dataloader(
        args.dataset,
        args.maxlen,
        args.batch_size,
        mode='train',
        num_workers=args.num_workers,
        num_negatives=args.num_negatives,
    )
    
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, _, _] = dataset
    
    # Create model
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    # Initialize weights
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.item_emb.weight.data[0, :] = 0
    
    # Apply graph embeddings
    if args.graph_emb_file and os.path.exists(args.graph_emb_file):
        initialize_sasrec_with_graph_embeddings(
            model,
            args.graph_emb_file,
            freeze_embeddings=False
        )
    
    # Apply pattern-aware initialization with current alpha
    if args.pattern_file and os.path.exists(args.pattern_file):
        pattern_initializer = PatternAwareInitializer(args.pattern_file)
        pattern_initializer.initialize_embeddings(model.item_emb, alpha=alpha)
        
        # Create pattern regularizer with current weight
        pattern_regularizer = PatternRegularizer(args.pattern_file, top_k=500)
    else:
        pattern_regularizer = None
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), weight_decay=0.01)
    
    # Training loop
    best_val_ndcg = 0.0
    best_val_hr = 0.0
    best_test_ndcg = 0.0
    best_test_hr = 0.0
    best_epoch = 0
    
    # Create log directory
    log_dir = os.path.join('tuning', stage_name)
    os.makedirs(log_dir, exist_ok=True)
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        
        epoch_rec_loss = 0.0
        epoch_pattern_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.num_epochs}", 
                   unit="batch", ncols=100, leave=False)
        
        for step, batch in enumerate(pbar):
            u, seq, pos, neg = [x.to(args.device) for x in batch]
            optimizer.zero_grad()
            
            mask = (pos != 0)
            mask_exp = mask.unsqueeze(-1).expand(-1, -1, args.num_negatives)
            
            pos_logits, neg_logits = model(u, seq, pos, neg)
            
            pos_sel = torch.masked_select(pos_logits, mask)
            
            if neg_logits.dim() == 2:
                neg_sel = torch.masked_select(neg_logits, mask).unsqueeze(1)
            else:
                neg_sel = torch.masked_select(neg_logits, mask_exp).view(-1, args.num_negatives)
            
            cand_logits = torch.cat([pos_sel.unsqueeze(1), neg_sel], dim=1)
            cand_labels = torch.zeros_like(cand_logits)
            cand_labels[:, 0] = 1.0
            
            rec_loss = gbce_loss(cand_logits, cand_labels)
            
            pattern_loss = torch.tensor(0.0, device=args.device)
            if pattern_regularizer:
                pattern_loss = pattern_regularizer.compute_loss(model.item_emb, args.device)
            
            total_loss = rec_loss + reg_weight * pattern_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_rec_loss += rec_loss.item()
            epoch_pattern_loss += pattern_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'rec': f'{rec_loss.item():.4f}',
                'pat': f'{pattern_loss.item():.4f}'
            })
        
        avg_rec_loss = epoch_rec_loss / max(1, num_batches)
        avg_pattern_loss = epoch_pattern_loss / max(1, num_batches)
        
        # Evaluation every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
            
            val_ndcg, val_hr = t_valid
            test_ndcg, test_hr = t_test
            
            print(f'Epoch {epoch:3d} | Rec: {avg_rec_loss:.4f} | Pat: {avg_pattern_loss:.4f}')
            print(f'         Val: NDCG={val_ndcg:.4f}, HR={val_hr:.4f}')
            print(f'         Test: NDCG={test_ndcg:.4f}, HR={test_hr:.4f}')
            
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_val_hr = val_hr
                best_test_ndcg = test_ndcg
                best_test_hr = test_hr
                best_epoch = epoch
                print(f'         ✓ New best!')
    
    results = {
        'stage_name': stage_name,
        'alpha': alpha,
        'reg_weight': reg_weight,
        'best_epoch': best_epoch,
        'best_val_ndcg': float(best_val_ndcg),
        'best_val_hr': float(best_val_hr),
        'best_test_ndcg': float(best_test_ndcg),
        'best_test_hr': float(best_test_hr),
    }
    
    # Save results
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Best Val NDCG: {best_val_ndcg:.4f} @ epoch {best_epoch}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=500, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=100, type=int,
                       help='Epochs per config (default: 100 for fast tuning)')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--num_negatives', default=15, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--norm_first', action='store_true', default=False)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pattern_file', type=str, required=True)
    parser.add_argument('--graph_emb_file', type=str, required=True)
    parser.add_argument('--seed', default=42, type=int)
    
    args = parser.parse_args()
    
    # Create tuning directory
    os.makedirs('tuning', exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed} (fixed for all experiments)")
    print(f"Epochs per config: {args.num_epochs}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    # ========================================
    # STAGE 1: Tune Alpha (fix reg_weight=0.01)
    # ========================================
    print(f"\n{'🔵'*40}")
    print(f"STAGE 1: TUNING ALPHA (reg_weight=0.01 fixed)")
    print(f"{'🔵'*40}\n")
    
    alpha_values = [0.2, 0.3, 0.4, 0.5]
    fixed_reg_weight = 0.01
    
    alpha_results = []
    for alpha in alpha_values:
        stage_name = f'alpha_{alpha:.1f}_reg_{fixed_reg_weight}'
        result = train_with_config(args, alpha, fixed_reg_weight, stage_name)
        alpha_results.append(result)
        all_results.append(result)
    
    # Find best alpha
    best_alpha_result = max(alpha_results, key=lambda x: x['best_val_ndcg'])
    best_alpha = best_alpha_result['alpha']
    
    print(f"\n{'='*80}")
    print(f"STAGE 1 RESULTS: Best Alpha = {best_alpha}")
    print(f"{'='*80}")
    for r in alpha_results:
        marker = "✅" if r['alpha'] == best_alpha else "  "
        print(f"{marker} Alpha={r['alpha']:.1f}: Val NDCG={r['best_val_ndcg']:.4f}, "
              f"Test NDCG={r['best_test_ndcg']:.4f}")
    print(f"{'='*80}\n")
    
    # ========================================
    # STAGE 2: Tune Reg Weight (use best alpha)
    # ========================================
    print(f"\n{'🟢'*40}")
    print(f"STAGE 2: TUNING REG_WEIGHT (alpha={best_alpha} fixed)")
    print(f"{'🟢'*40}\n")
    
    reg_weight_values = [0.005, 0.01, 0.02, 0.05]
    
    reg_results = []
    for reg_weight in reg_weight_values:
        stage_name = f'alpha_{best_alpha:.1f}_reg_{reg_weight}'
        result = train_with_config(args, best_alpha, reg_weight, stage_name)
        reg_results.append(result)
        all_results.append(result)
    
    # Find best reg_weight
    best_reg_result = max(reg_results, key=lambda x: x['best_val_ndcg'])
    best_reg_weight = best_reg_result['reg_weight']
    
    print(f"\n{'='*80}")
    print(f"STAGE 2 RESULTS: Best Reg Weight = {best_reg_weight}")
    print(f"{'='*80}")
    for r in reg_results:
        marker = "✅" if r['reg_weight'] == best_reg_weight else "  "
        print(f"{marker} Reg={r['reg_weight']:.3f}: Val NDCG={r['best_val_ndcg']:.4f}, "
              f"Test NDCG={r['best_test_ndcg']:.4f}")
    print(f"{'='*80}\n")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    best_overall = max(all_results, key=lambda x: x['best_val_ndcg'])
    
    print(f"\n{'🏆'*40}")
    print(f"FINAL BEST HYPERPARAMETERS")
    print(f"{'🏆'*40}")
    print(f"Alpha: {best_overall['alpha']:.2f}")
    print(f"Reg Weight: {best_overall['reg_weight']:.4f}")
    print(f"Val NDCG@10: {best_overall['best_val_ndcg']:.4f}")
    print(f"Test NDCG@10: {best_overall['best_test_ndcg']:.4f}")
    print(f"Test HR@10: {best_overall['best_test_hr']:.4f}")
    print(f"Best Epoch: {best_overall['best_epoch']}")
    print(f"{'🏆'*40}\n")
    
    # Save all results
    summary = {
        'best_alpha': best_alpha,
        'best_reg_weight': best_reg_weight,
        'best_config': best_overall,
        'all_results': all_results
    }
    
    with open('tuning/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ All results saved to: tuning/summary.json")
    print(f"\nNext step: Train full model with best hyperparameters:")
    print(f"  python benchmark_configs.py \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --pattern_init_alpha {best_overall['alpha']} \\")
    print(f"    --pattern_reg_weight {best_overall['reg_weight']} \\")
    print(f"    --num_epochs 600")
    print()


if __name__ == '__main__':
    main()
