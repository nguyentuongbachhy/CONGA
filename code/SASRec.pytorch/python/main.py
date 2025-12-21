import os
import time

import torch
import argparse
import numpy as np

from model import SASRec
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


parser = argparse.ArgumentParser()
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
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers')

args = parser.parse_args()

# Create output directory
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))


if __name__ == '__main__':
    # Check and convert dataset to binary format if needed
    check_and_convert_dataset(args.dataset)
    
    # Load metadata
    usernum, itemnum = load_metadata(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Users: {usernum}, Items: {itemnum}")
    
    # Create DataLoader
    train_loader = get_dataloader(
        args.dataset, 
        args.maxlen, 
        args.batch_size, 
        mode='train',
        num_workers=args.num_workers
    )
    
    # Calculate average sequence length
    print(f"Training batches per epoch: {len(train_loader)}")
    
    # For evaluation, we still use the legacy data_partition
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, _, _] = dataset
    
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print(f'Average sequence length: {cc / len(user_train):.2f}')
    
    # Setup logging
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    # Initialize model
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    # Initialize weights
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass  # Ignore failed initializations (e.g., 1D tensors)
    
    # Set padding embeddings to zero
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    
    # Load checkpoint if provided
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f"Loaded checkpoint from epoch {epoch_start_idx - 1}")
        except Exception as e:
            print(f'Failed loading state_dicts from: {args.state_dict_path}')
            print(f'Error: {e}')
            import pdb
            pdb.set_trace()
    
    # Inference only mode
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
        exit(0)
    
    # Setup training
    model.train()
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    # AMP: Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training metrics
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    
    print("\n" + "="*60)
    print("Starting Training with AMP (Automatic Mixed Precision)")
    print("="*60 + "\n")
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            u, seq, pos, neg = batch
            u = u.numpy()
            seq = seq.numpy()
            pos = pos.numpy()
            neg = neg.numpy()
            
            adam_optimizer.zero_grad()
            
            # AMP: Use autocast for forward pass
            with torch.amp.autocast('cuda'):
                pos_logits, neg_logits = model(u, seq, pos, neg)
                
                # Create labels
                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)
                
                # Compute loss only on non-padding positions
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                
                # L2 regularization on item embeddings
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.sum(param ** 2)
            
            # AMP: Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(adam_optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if (step + 1) % 100 == 0 or step == 0:
                print(f"Epoch {epoch}, Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        # Evaluation every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            
            print('Evaluating', end='')
            with torch.no_grad():
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
            
            print(f'\nEpoch: {epoch}, Time: {T:.1f}s, '
                  f'Valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), '
                  f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
            
            # Save best model
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or \
               t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                
                folder = args.dataset + '_' + args.train_dir
                fname = f'SASRec.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
                torch.save(model.state_dict(), os.path.join(folder, fname))
                print(f"Model saved: {fname}")
            
            f.write(f'{epoch} {t_valid} {t_test}\n')
            f.flush()
            t0 = time.time()
    
    # Save final model
    folder = args.dataset + '_' + args.train_dir
    fname = f'SASRec.epoch={args.num_epochs}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
    torch.save(model.state_dict(), os.path.join(folder, fname))
    print(f"\nFinal model saved: {fname}")
    
    f.close()
    print("Training completed!")

