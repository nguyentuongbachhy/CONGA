import os
import time
from typing import Any, Tuple

import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from model import SASRec
from graph_teacher import LightGCN
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
parser.add_argument('--num_negatives', default=1, type=int, help='Number of negatives per position')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--use_amp', default=True, type=str2bool)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers')
parser.add_argument('--use_distillation', default=False, type=str2bool, help='Use graph distillation')
parser.add_argument('--teacher_path', default=None, type=str, help='Path to teacher model checkpoint')
parser.add_argument('--distill_weight', default=0.5, type=float, help='Weight for distillation loss')
parser.add_argument('--teacher_dim', default=64, type=int, help='Teacher embedding dimension')
parser.add_argument('--teacher_layers', default=3, type=int, help='Teacher num layers')
parser.add_argument('--distill_temp', default=1.0, type=float, help='Temperature for distillation')

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
        num_workers=args.num_workers,
        num_negatives=args.num_negatives,
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
    
    teacher: Any = None
    if args.use_distillation:
        if args.teacher_path is None:
            print('Error: --teacher_path required when --use_distillation is true')
            exit(1)
        print(f'Loading teacher model from {args.teacher_path}')
        teacher = LightGCN(
            num_users=usernum,
            num_items=itemnum,
            embedding_dim=args.teacher_dim,
            num_layers=args.teacher_layers,
            device=args.device,
        ).to(args.device)
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=torch.device(args.device)))
        teacher.Graph = teacher.build_graph(user_train)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        print('Teacher model loaded and frozen')
    
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
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )

    use_amp: bool = bool(args.use_amp) and str(args.device).startswith('cuda')
    scaler: Any = torch.amp.grad_scaler.GradScaler(device='cuda') if use_amp else None
    
    best_val_ndcg: float = 0.0
    best_val_hr: float = 0.0
    best_test_ndcg: float = 0.0
    best_test_hr: float = 0.0
    T: float = 0.0
    t0: float = time.time()
    
    print("\n" + "="*70)
    if use_amp:
        print("Starting Training with AMP (Automatic Mixed Precision)")
    else:
        print("Starting Training (FP32)")
    print("="*70)
    print(f"Model: SASRec-RoPE | Dataset: {args.dataset}")
    print(f"Hidden: {args.hidden_units} | Layers: {args.num_blocks} | Heads: {args.num_heads}")
    print(f"Max Length: {args.maxlen} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("="*70 + "\n")
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        epoch_loss: float = 0.0
        num_batches: int = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.num_epochs}", unit="batch", ncols=100)
        for step, batch in enumerate(pbar):
            u: np.ndarray
            seq: np.ndarray
            pos: np.ndarray
            neg: np.ndarray
            u, seq, pos, neg = batch
            
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast_mode.autocast(device_type='cuda'):
                    pos_logits, neg_logits = model(u, seq, pos, neg)
                    indices: Tuple[np.ndarray, ...] = np.where(pos != 0)

                    pos_selected: torch.Tensor = pos_logits[indices]  # [M]
                    neg_selected: torch.Tensor = neg_logits[indices]  # [M] or [M, N]
                    if neg_selected.dim() == 1:
                        neg_selected = neg_selected.unsqueeze(1)
                    cand_logits: torch.Tensor = torch.cat(
                        [pos_selected.unsqueeze(1), neg_selected],
                        dim=1,
                    )  # [M, 1+N]
                    loss: torch.Tensor = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()

                    if args.use_distillation and teacher is not None:
                        with torch.no_grad():
                            pos_t = torch.as_tensor(pos, dtype=torch.long, device=args.device)
                            neg_t = torch.as_tensor(neg, dtype=torch.long, device=args.device)
                            batch_users = torch.as_tensor(u, dtype=torch.long, device=args.device)
                            
                            teacher_pos_emb = teacher.get_item_embedding(pos_t)
                            teacher_neg_emb = teacher.get_item_embedding(neg_t)
                            user_emb = teacher.get_user_embedding(batch_users)
                            
                            teacher_pos_logits = (user_emb.unsqueeze(1) * teacher_pos_emb).sum(dim=-1)
                            if teacher_neg_emb.dim() == 3:
                                teacher_neg_logits = (user_emb.unsqueeze(1).unsqueeze(2) * teacher_neg_emb).sum(dim=-1)
                            else:
                                teacher_neg_logits = (user_emb.unsqueeze(1) * teacher_neg_emb).sum(dim=-1)
                        
                        teacher_pos_sel = teacher_pos_logits[indices]
                        teacher_neg_sel = teacher_neg_logits[indices]
                        if teacher_neg_sel.dim() == 1:
                            teacher_neg_sel = teacher_neg_sel.unsqueeze(1)
                        teacher_cand = torch.cat([teacher_pos_sel.unsqueeze(1), teacher_neg_sel], dim=1)
                        
                        distill_loss = F.kl_div(
                            F.log_softmax(cand_logits / args.distill_temp, dim=1),
                            F.softmax(teacher_cand / args.distill_temp, dim=1),
                            reduction='batchmean'
                        ) * (args.distill_temp ** 2)
                        
                        loss = loss + args.distill_weight * distill_loss

                    for param in model.item_emb.parameters():
                        loss += args.l2_emb * torch.sum(param ** 2)

                scaler.scale(loss).backward()
                scaler.step(optimizer=optimizer)
                scaler.update()
            else:
                pos_logits, neg_logits = model(u, seq, pos, neg)
                indices: Tuple[np.ndarray, ...] = np.where(pos != 0)

                pos_selected: torch.Tensor = pos_logits[indices]  # [M]
                neg_selected: torch.Tensor = neg_logits[indices]  # [M] or [M, N]
                if neg_selected.dim() == 1:
                    neg_selected = neg_selected.unsqueeze(1)
                cand_logits: torch.Tensor = torch.cat(
                    [pos_selected.unsqueeze(1), neg_selected],
                    dim=1,
                )  # [M, 1+N]
                loss: torch.Tensor = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()

                if args.use_distillation and teacher is not None:
                    with torch.no_grad():
                        pos_t = torch.as_tensor(pos, dtype=torch.long, device=args.device)
                        neg_t = torch.as_tensor(neg, dtype=torch.long, device=args.device)
                        batch_users = torch.as_tensor(u, dtype=torch.long, device=args.device)
                        
                        teacher_pos_emb = teacher.get_item_embedding(pos_t)
                        teacher_neg_emb = teacher.get_item_embedding(neg_t)
                        user_emb = teacher.get_user_embedding(batch_users)
                        
                        teacher_pos_logits = (user_emb.unsqueeze(1) * teacher_pos_emb).sum(dim=-1)
                        if teacher_neg_emb.dim() == 3:
                            teacher_neg_logits = (user_emb.unsqueeze(1).unsqueeze(2) * teacher_neg_emb).sum(dim=-1)
                        else:
                            teacher_neg_logits = (user_emb.unsqueeze(1) * teacher_neg_emb).sum(dim=-1)
                    
                    teacher_pos_sel = teacher_pos_logits[indices]
                    teacher_neg_sel = teacher_neg_logits[indices]
                    if teacher_neg_sel.dim() == 1:
                        teacher_neg_sel = teacher_neg_sel.unsqueeze(1)
                    teacher_cand = torch.cat([teacher_pos_sel.unsqueeze(1), teacher_neg_sel], dim=1)
                    
                    distill_loss = F.kl_div(
                        F.log_softmax(cand_logits / args.distill_temp, dim=1),
                        F.softmax(teacher_cand / args.distill_temp, dim=1),
                        reduction='batchmean'
                    ) * (args.distill_temp ** 2)
                    
                    loss = loss + args.distill_weight * distill_loss

                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.sum(param ** 2)

                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{epoch_loss/num_batches:.4f}'})
        
        avg_loss: float = epoch_loss / len(train_loader)
        
        print(f'Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}', end='')
        
        if epoch % 20 == 0:
            model.eval()
            t1: float = time.time() - t0
            T += t1
            
            with torch.no_grad():
                # Ensure evaluation is always FP32 (no autocast)
                with torch.amp.autocast_mode.autocast(device_type='cuda', enabled=False):
                    t_test = evaluate(model, dataset, args)
                    t_valid = evaluate_valid(model, dataset, args)
            
            print(f' | Time: {T:.1f}s')
            print(f'         Valid - NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}')
            print(f'         Test  - NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f}')
            
            is_best: bool = False
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or \
               t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                is_best = True
                
                folder: str = args.dataset + '_' + args.train_dir
                fname: str = f'SASRec.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
                torch.save(model.state_dict(), os.path.join(folder, fname))
                print(f'         ✓ New best model saved: {fname}')
            
            if is_best:
                print(f'         Best so far - Valid NDCG: {best_val_ndcg:.4f}, Test NDCG: {best_test_ndcg:.4f}')
            
            f.write(f'{epoch} {t_valid} {t_test}\n')
            f.flush()
            t0 = time.time()
            print()
        else:
            print()
    
    folder = args.dataset + '_' + args.train_dir
    fname = f'SASRec.epoch={args.num_epochs}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
    torch.save(model.state_dict(), os.path.join(folder, fname))
    print(f"\nFinal model saved: {fname}")
    
    f.close()
    print("Training completed!")

