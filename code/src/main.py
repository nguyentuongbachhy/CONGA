import os
import time
from typing import Any, Tuple

import torch
import argparse
import numpy as np
from tqdm import tqdm

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
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f_args:
    f_args.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))


if __name__ == '__main__':
    check_and_convert_dataset(args.dataset)
    
    usernum: int
    itemnum: int
    usernum, itemnum = load_metadata(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Users: {usernum}, Items: {itemnum}")
    
    train_loader: Any = get_dataloader(
        args.dataset, 
        args.maxlen, 
        args.batch_size, 
        mode='train',
        num_workers=args.num_workers
    )
    
    print(f"Training batches per epoch: {len(train_loader)}")
    
    dataset: Tuple[Any, Any, Any, Any, Any] = data_partition(args.dataset)
    [user_train, user_valid, user_test, _, _] = dataset
    
    cc: float = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print(f'Average sequence length: {cc / len(user_train):.2f}')
    
    f: Any = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    model: SASRec = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.item_emb.weight.data[0, :] = 0
    
    epoch_start_idx: int = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail: str = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f"Loaded checkpoint from epoch {epoch_start_idx - 1}")
        except Exception as e:
            print(f'Failed loading state_dicts from: {args.state_dict_path}')
            print(f'Error: {e}')
            import pdb
            pdb.set_trace()
    
    if args.inference_only:
        model.eval()
        t_test: Tuple[float, float] = evaluate(model, dataset, args)
        print(f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
        exit(0)
    
    model.train()
    bce_criterion: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )
    
    scaler: torch.amp.grad_scaler.GradScaler = torch.amp.grad_scaler.GradScaler(device='cuda')
    
    best_val_ndcg: float = 0.0
    best_val_hr: float = 0.0
    best_test_ndcg: float = 0.0
    best_test_hr: float = 0.0
    T: float = 0.0
    t0: float = time.time()
    
    print("\n" + "="*60)
    print("Starting Training with AMP (Automatic Mixed Precision)")
    print("="*60 + "\n")
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        epoch_loss: float = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", unit="batch")
        for step, batch in enumerate(pbar):
            u: np.ndarray
            seq: np.ndarray
            pos: np.ndarray
            neg: np.ndarray
            u, seq, pos, neg = batch
            
            optimizer.zero_grad()
            
            with torch.amp.autocast_mode.autocast(device_type='cuda'):
                pos_logits: torch.Tensor
                neg_logits: torch.Tensor
                pos_logits, neg_logits = model(u, seq, pos, neg)
                
                pos_labels: torch.Tensor = torch.ones_like(pos_logits)
                neg_labels: torch.Tensor = torch.zeros_like(neg_logits)
                
                indices: Tuple[np.ndarray, ...] = np.where(pos != 0)
                loss: torch.Tensor = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.sum(param ** 2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss: float = epoch_loss / len(train_loader)
        
        if epoch % 20 == 0:
            model.eval()
            t1: float = time.time() - t0
            T += t1
            
            with torch.no_grad():
                t_test: Tuple[float, float] = evaluate(model, dataset, args)
                t_valid: Tuple[float, float] = evaluate_valid(model, dataset, args)
            
            print(f'\nEpoch: {epoch}, Time: {T:.1f}s, '
                  f'Valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), '
                  f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
            
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or \
               t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                
                folder: str = args.dataset + '_' + args.train_dir
                fname: str = f'SASRec.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
                torch.save(model.state_dict(), os.path.join(folder, fname))
                print(f"Model saved: {fname}")
            
            f.write(f'{epoch} {t_valid} {t_test}\n')
            f.flush()
            t0 = time.time()
    
    folder = args.dataset + '_' + args.train_dir
    fname = f'SASRec.epoch={args.num_epochs}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
    torch.save(model.state_dict(), os.path.join(folder, fname))
    print(f"\nFinal model saved: {fname}")
    
    f.close()
    print("Training completed!")

