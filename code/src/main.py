import os
import time

import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from model import SASRec
# from graph_teacher import LightGCN
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
parser.add_argument('--num_negatives', default=1, type=int, help='Number of negatives per position')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--use_amp', default=True, type=str2bool)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers')

args = parser.parse_args()

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f_args:
    f_args.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

if __name__ == '__main__':
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    
    tensor_mem_gb = (args.batch_size * args.maxlen * args.num_negatives * args.hidden_units * 4) / (1024**3)
    print(f"Dataset: {args.dataset} | Users: {usernum} | Items: {itemnum}")
    print(f"Estimated VRAM for Negatives Tensor: {tensor_mem_gb:.2f} GB")
    
    if tensor_mem_gb > 8.0:
        print("\n[WARNING] Cấu hình này yêu cầu VRAM rất lớn!")
        print("Gợi ý: Giảm --num_negatives (ví dụ: 1) hoặc giảm --batch_size.\n")

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
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.item_emb.weight.data[0, :] = 0
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except Exception as e:
            print(f'Failed loading state_dicts: {e}')
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
        exit(0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    
    use_amp = bool(args.use_amp) and str(args.device).startswith('cuda')
    scaler = torch.amp.grad_scaler.GradScaler(device='cuda') if use_amp else None
    
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T, t0 = 0.0, time.time()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        epoch_loss, num_batches = 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", unit="batch", ncols=100)
        
        for step, (u, seq, pos) in enumerate(pbar):
            u, seq, pos = [x.to(args.device) for x in [u, seq, pos]]

            neg = torch.randint(1, itemnum + 1, (u.size(0), args.maxlen, args.num_negatives), device=args.device)

            collision = (neg == pos.unsqueeze(-1))
            neg[collision] = (neg[collision] % itemnum) + 1
            optimizer.zero_grad()
            mask = (pos != 0) 

            with torch.amp.autocast_mode.autocast(device_type='cuda', enabled=use_amp):
                pos_logits, neg_logits = model(u, seq, pos, neg)

                logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)

                targets = torch.zeros((logits.size(0), logits.size(1)), dtype=torch.long, device=args.device)

                loss_per_token = loss_fn(logits.view(-1, 1 + args.num_negatives), targets.view(-1))
                
                loss_per_token = loss_per_token.view(logits.size(0), logits.size(1))
                
                loss = (loss_per_token * mask).sum() / mask.sum()

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / max(1, num_batches)
        print(f'Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}', end='')
        
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
                folder = args.dataset + '_' + args.train_dir
                fname = f'SASRec.best.pth'
                torch.save(model.state_dict(), os.path.join(folder, fname))
                print(f'         ✓ Saved best model')
            
            f.write(f'{epoch} {t_valid} {t_test}\n')
            f.flush()
            t0 = time.time()
        else:
            print()
            
    f.close()
    print("Training completed!")