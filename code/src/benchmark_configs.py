"""
Benchmark script to compare 3 configurations with fixed seed for fair comparison:
1. Baseline: gbce loss only
2. +Patterns: gbce + pattern-aware initialization & regularization
3. +Patterns+Graph: gbce + patterns + graph pretrained embeddings

All experiments use the same config and random seed for fair comparison.
"""

import os
import sys
import time
import json
import argparse
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
    """
    Generalized Binary Cross-Entropy (gBCE) loss for recommendation systems.
    Reference: "gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling" (2023)
    """
    y_pred_scaled = y_pred / temperature
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_loss = -(y_true * torch.log(sigmoid_pred + eps) +
                 (1 - y_true) * torch.log(1 - sigmoid_pred + eps))
    bce_loss = torch.mean(bce_loss)
    ce_loss = -F.log_softmax(y_pred_scaled, dim=1)[:, 0]
    ce_loss = torch.mean(ce_loss)
    total_loss = alpha * bce_loss + (1 - alpha) * ce_loss
    return total_loss


def train_one_config(config_name: str, args, use_patterns: bool = False, 
                     use_graph: bool = False, pattern_file: str = None,
                     graph_emb_file: str = None):
    """
    Train one configuration and return results
    
    Args:
        config_name: Name of configuration (for logging)
        args: Training arguments
        use_patterns: Whether to use pattern-aware initialization and regularization
        use_graph: Whether to use graph pretrained embeddings
        pattern_file: Path to patterns.pkl file
        graph_emb_file: Path to graph embeddings file
    """
    print(f"\n{'='*80}")
    print(f"CONFIGURATION: {config_name}")
    print(f"{'='*80}")
    print(f"Patterns: {use_patterns} | Graph: {use_graph}")
    print(f"Loss: gbce | Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load data
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    
    # Create dataloaders
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
    
    # Apply graph embeddings if specified
    if use_graph and graph_emb_file and os.path.exists(graph_emb_file):
        print(f"🔷 Loading graph embeddings from {graph_emb_file}")
        initialize_sasrec_with_graph_embeddings(
            model,
            graph_emb_file,
            freeze_embeddings=False
        )
        print(f"✓ Graph embeddings loaded\n")
    
    # Apply pattern-aware initialization if specified
    pattern_initializer = None
    pattern_regularizer = None
    
    if use_patterns and pattern_file and os.path.exists(pattern_file):
        print(f"🎯 Loading patterns from {pattern_file}")
        pattern_initializer = PatternAwareInitializer(pattern_file)
        pattern_initializer.initialize_embeddings(
            model.item_emb,
            alpha=args.pattern_init_alpha
        )
        print(f"✓ Pattern-aware initialization applied\n")
        
        # Create pattern regularizer
        pattern_regularizer = PatternRegularizer(pattern_file, top_k=500)
        print(f"✓ Pattern regularizer created\n")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), weight_decay=0.01)
    
    # Training loop
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    best_epoch = 0
    T, t0 = 0.0, time.time()
    
    # Create log directory
    log_dir = os.path.join('benchmark', config_name)
    os.makedirs(log_dir, exist_ok=True)
    
    f = open(os.path.join(log_dir, 'log.txt'), 'w')
    f.write('epoch,val_ndcg,val_hr,test_ndcg,test_hr,rec_loss,pattern_loss,time\n')
    
    print(f"Training for {args.num_epochs} epochs...")
    print(f"Results will be saved to: {log_dir}\n")
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        
        epoch_rec_loss = 0.0
        epoch_pattern_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.num_epochs}", 
                   unit="batch", ncols=120)
        
        for step, batch in enumerate(pbar):
            u, seq, pos, neg = [x.to(args.device) for x in batch]
            optimizer.zero_grad()
            
            # Compute recommendation loss
            mask = (pos != 0)
            mask_exp = mask.unsqueeze(-1).expand(-1, -1, args.num_negatives)
            
            pos_logits, neg_logits = model(u, seq, pos, neg)
            
            pos_sel = torch.masked_select(pos_logits, mask)
            
            if neg_logits.dim() == 2:
                neg_sel = torch.masked_select(neg_logits, mask).unsqueeze(1)
            else:
                neg_sel = torch.masked_select(neg_logits, mask_exp).view(-1, args.num_negatives)
            
            cand_logits = torch.cat([pos_sel.unsqueeze(1), neg_sel], dim=1)
            
            # Create labels for gbce
            cand_labels = torch.zeros_like(cand_logits)
            cand_labels[:, 0] = 1.0
            
            # Compute gbce loss
            rec_loss = gbce_loss(cand_logits, cand_labels)
            
            # Pattern regularization loss
            pattern_loss = torch.tensor(0.0, device=args.device)
            if pattern_regularizer:
                pattern_loss = pattern_regularizer.compute_loss(model.item_emb, args.device)
            
            # Total loss
            total_loss = rec_loss + args.pattern_reg_weight * pattern_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_rec_loss += rec_loss.item()
            epoch_pattern_loss += pattern_loss.item()
            num_batches += 1
            
            # Update progress bar
            if pattern_regularizer:
                pbar.set_postfix({
                    'rec': f'{rec_loss.item():.4f}',
                    'pat': f'{pattern_loss.item():.4f}'
                })
            else:
                pbar.set_postfix({'loss': f'{rec_loss.item():.4f}'})
        
        avg_rec_loss = epoch_rec_loss / max(1, num_batches)
        avg_pattern_loss = epoch_pattern_loss / max(1, num_batches)
        
        # Evaluation
        if epoch % 20 == 0:
            model.eval()
            T += time.time() - t0
            with torch.no_grad():
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
            
            val_ndcg, val_hr = t_valid
            test_ndcg, test_hr = t_test
            
            print(f'Epoch {epoch:3d} | Rec: {avg_rec_loss:.4f} | Pat: {avg_pattern_loss:.4f} | Time: {T:.1f}s')
            print(f'         Valid: NDCG@10={val_ndcg:.4f}, HR@10={val_hr:.4f}')
            print(f'         Test:  NDCG@10={test_ndcg:.4f}, HR@10={test_hr:.4f}')
            
            if val_ndcg > best_val_ndcg:
                best_val_ndcg, best_val_hr = val_ndcg, val_hr
                best_test_ndcg, best_test_hr = test_ndcg, test_hr
                best_epoch = epoch
                
                # Save checkpoint
                fname = 'best_model.pth'
                torch.save(model.state_dict(), os.path.join(log_dir, fname))
                print(f'         ✓ Saved best model (epoch {epoch})')
            
            f.write(f'{epoch},{val_ndcg:.6f},{val_hr:.6f},{test_ndcg:.6f},{test_hr:.6f},'
                   f'{avg_rec_loss:.6f},{avg_pattern_loss:.6f},{T:.2f}\n')
            f.flush()
            t0 = time.time()
    
    f.close()
    
    # Save final results
    results = {
        'config_name': config_name,
        'use_patterns': use_patterns,
        'use_graph': use_graph,
        'best_epoch': best_epoch,
        'best_val_ndcg': float(best_val_ndcg),
        'best_val_hr': float(best_val_hr),
        'best_test_ndcg': float(best_test_ndcg),
        'best_test_hr': float(best_test_hr),
        'total_time': float(T),
        'seed': args.seed,
    }
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {config_name}")
    print(f"{'='*80}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val NDCG@10: {best_val_ndcg:.4f} | HR@10: {best_val_hr:.4f}")
    print(f"Best Test NDCG@10: {best_test_ndcg:.4f} | HR@10: {best_test_hr:.4f}")
    print(f"Total Time: {T:.1f}s")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark 3 Configurations')
    
    # Basic arguments
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=400, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--num_negatives', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--norm_first', action='store_true', default=False)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # Pattern and graph files
    parser.add_argument('--pattern_file', type=str, default=None)
    parser.add_argument('--graph_emb_file', type=str, default=None)
    
    # Pattern parameters
    parser.add_argument('--pattern_init_alpha', default=0.3, type=float)
    parser.add_argument('--pattern_reg_weight', default=0.01, type=float)
    
    # Seed for reproducibility
    parser.add_argument('--seed', default=42, type=int,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create benchmark directory
    os.makedirs('benchmark', exist_ok=True)
    
    # Save config
    config_dict = vars(args)
    with open('benchmark/config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK: 3 CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed} (fixed for fair comparison)")
    print(f"Epochs: {args.num_epochs}")
    print(f"Loss: gbce")
    print(f"{'='*80}\n")
    
    all_results = []
    
    # Configuration 1: Baseline (gbce only)
    print("\n" + "🔵"*40)
    print("CONFIGURATION 1: Baseline (gbce only)")
    print("🔵"*40)
    results_1 = train_one_config(
        config_name='1_baseline_gbce',
        args=args,
        use_patterns=False,
        use_graph=False,
        pattern_file=None,
        graph_emb_file=None
    )
    all_results.append(results_1)
    
    # Configuration 2: gbce + Patterns
    print("\n" + "🟢"*40)
    print("CONFIGURATION 2: gbce + Patterns")
    print("🟢"*40)
    results_2 = train_one_config(
        config_name='2_gbce_patterns',
        args=args,
        use_patterns=True,
        use_graph=False,
        pattern_file=args.pattern_file,
        graph_emb_file=None
    )
    all_results.append(results_2)
    
    # Configuration 3: gbce + Patterns + Graph
    print("\n" + "🟣"*40)
    print("CONFIGURATION 3: gbce + Patterns + Graph")
    print("🟣"*40)
    results_3 = train_one_config(
        config_name='3_gbce_patterns_graph',
        args=args,
        use_patterns=True,
        use_graph=True,
        pattern_file=args.pattern_file,
        graph_emb_file=args.graph_emb_file
    )
    all_results.append(results_3)
    
    # Save comparison results
    with open('benchmark/comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"{'Config':<30} {'Val NDCG@10':<15} {'Test NDCG@10':<15} {'Test HR@10':<15}")
    print("-"*80)
    for result in all_results:
        print(f"{result['config_name']:<30} "
              f"{result['best_val_ndcg']:<15.4f} "
              f"{result['best_test_ndcg']:<15.4f} "
              f"{result['best_test_hr']:<15.4f}")
    print("="*80)
    
    # Calculate improvements
    baseline_ndcg = all_results[0]['best_test_ndcg']
    print(f"\nImprovements over baseline:")
    for i, result in enumerate(all_results[1:], 1):
        improvement = ((result['best_test_ndcg'] - baseline_ndcg) / baseline_ndcg) * 100
        print(f"  Config {i+1}: {improvement:+.2f}% NDCG@10")
    
    print(f"\n✓ All results saved to: benchmark/")
    print(f"  - benchmark/comparison.json")
    print(f"  - benchmark/1_baseline_gbce/")
    print(f"  - benchmark/2_gbce_patterns/")
    print(f"  - benchmark/3_gbce_patterns_graph/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
