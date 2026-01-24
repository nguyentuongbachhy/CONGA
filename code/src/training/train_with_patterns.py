import os
import time
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from models.model import SASRec
from pattern_mining.pattern_utils import (
    PatternAwareInitializer,
    PatternRegularizer,
    print_pattern_stats
)
from utils import (
    check_and_convert_dataset, 
    load_metadata, 
    get_dataloader, 
    data_partition,
    evaluate, 
    evaluate_valid
)


def str2bool(s: str) -> bool:
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def train_with_patterns(args) -> None:
    """
    Train SASRec with pattern-based improvements
    """
    
    # Load data
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    
    print(f"\n{'='*60}")
    print(f"TRAINING WITH PATTERNS: {args.dataset}")
    print(f"{'='*60}")
    print(f"Users: {usernum:,} | Items: {itemnum:,}")
    
    # Show pattern statistics
    if args.pattern_file and os.path.exists(args.pattern_file):
        print_pattern_stats(args.pattern_file)
    
    # Pattern-based components
    pattern_initializer = None
    pattern_regularizer = None
    
    if args.pattern_file and os.path.exists(args.pattern_file):
        print(f"\n🎯 Loading pattern-based components...")
        
        if args.use_pattern_init:
            pattern_initializer = PatternAwareInitializer(args.pattern_file)
        
        if args.use_pattern_reg:
            pattern_regularizer = PatternRegularizer(args.pattern_file, top_k=500)
    
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
    
    # Apply pattern-aware initialization
    if pattern_initializer and args.use_pattern_init:
        pattern_initializer.initialize_embeddings(
            model.item_emb, 
            alpha=args.pattern_init_alpha
        )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                  betas=(0.9, 0.98), weight_decay=0.01)
    
    # Training loop
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T, t0 = 0.0, time.time()
    
    # Create log file
    log_dir = args.dataset + '_' + args.train_dir
    os.makedirs(log_dir, exist_ok=True)
    f = open(os.path.join(log_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Pattern initialization: {args.use_pattern_init}")
    if args.use_pattern_init:
        print(f"  Alpha: {args.pattern_init_alpha}")
    print(f"Pattern regularization: {args.use_pattern_reg}")
    if args.use_pattern_reg:
        print(f"  Weight: {args.pattern_reg_weight}")
    print(f"Loss function: {args.loss_type}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        
        epoch_rec_loss = 0.0
        epoch_pattern_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", unit="batch", ncols=120)
        
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
            
            # Standard recommendation loss
            if args.loss_type == 'sampled_softmax':
                rec_loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()
            else:
                rec_loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()
            
            # Pattern regularization loss
            pattern_loss = torch.tensor(0.0, device=args.device)
            if pattern_regularizer and args.use_pattern_reg:
                pattern_loss = pattern_regularizer.compute_loss(model.item_emb, args.device)
            
            # Total loss
            total_loss = rec_loss + args.pattern_reg_weight * pattern_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_rec_loss += rec_loss.item()
            epoch_pattern_loss += pattern_loss.item()
            num_batches += 1
            
            # Update progress bar
            if args.use_pattern_reg:
                pbar.set_postfix({
                    'rec': f'{rec_loss.item():.4f}',
                    'pat': f'{pattern_loss.item():.4f}'
                })
            else:
                pbar.set_postfix({'loss': f'{rec_loss.item():.4f}'})
        
        avg_rec_loss = epoch_rec_loss / max(1, num_batches)
        avg_pattern_loss = epoch_pattern_loss / max(1, num_batches)
        
        if args.use_pattern_reg:
            print(f'Epoch {epoch:3d} | Rec: {avg_rec_loss:.4f} | Pattern: {avg_pattern_loss:.4f}', end='')
        else:
            print(f'Epoch {epoch:3d} | Loss: {avg_rec_loss:.4f}', end='')
        
        # Evaluation
        if epoch % 20 == 0:
            model.eval()
            T += time.time() - t0
            with torch.no_grad():
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
            
            print(f' | Time: {T:.1f}s')
            print(f'         Valid: {t_valid} | Test: {t_test}')
            
            if t_valid[0] > best_val_ndcg:
                best_val_ndcg, best_val_hr = t_valid
                best_test_ndcg, best_test_hr = t_test
                
                # Save checkpoint
                folder = log_dir
                fname = f'SASRec.best.pth'
                torch.save(model.state_dict(), os.path.join(folder, fname))
                print(f'         ✓ Saved best model')
            
            f.write(f'{epoch} {t_valid} {t_test}\n')
            f.flush()
            t0 = time.time()
        else:
            print()
    
    f.close()
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation NDCG@10: {best_val_ndcg:.4f}")
    print(f"Best test NDCG@10: {best_test_ndcg:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SASRec with Pattern-based Improvements')
    
    # Basic arguments
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--num_negatives', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--norm_first', action='store_true', default=False)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--loss_type', default='sampled_softmax', type=str)
    
    # Pattern-based arguments
    parser.add_argument('--pattern_file', type=str, default=None,
                        help='Path to patterns.pkl file (e.g., pattern_data/ml-1m_patterns.pkl)')
    parser.add_argument('--use_pattern_init', default=True, type=str2bool,
                        help='Use pattern-aware initialization (default: True)')
    parser.add_argument('--pattern_init_alpha', default=0.3, type=float,
                        help='Weight for pattern initialization (0=random, 1=full pattern, default: 0.3)')
    parser.add_argument('--use_pattern_reg', default=True, type=str2bool,
                        help='Use pattern regularization loss (default: True)')
    parser.add_argument('--pattern_reg_weight', default=0.01, type=float,
                        help='Weight for pattern regularization loss (default: 0.01)')
    
    args = parser.parse_args()
    
    # Validate pattern file
    if args.pattern_file and not os.path.exists(args.pattern_file):
        print(f"\nWarning: Pattern file not found: {args.pattern_file}")
        print(f"   Please run: python mine_patterns.py --dataset {args.dataset}")
        print(f"   Continuing without patterns...\n")
        args.pattern_file = None
        args.use_pattern_init = False
        args.use_pattern_reg = False
    
    train_with_patterns(args)
