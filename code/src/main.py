import os
import time

import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from model import SASRec
# from graph_teacher import LightGCN
from continuum_memory import ContinuumItemEmbedding
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
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers')
parser.add_argument('--use_nested_learning', default=False, type=str2bool,
                    help='Enable Continuum Memory System (CMS) for multi-timescale learning')
parser.add_argument('--cms_fast_weight', default=0.5, type=float,
                    help='Weight for fast memory in CMS (recent interactions)')
parser.add_argument('--cms_medium_weight', default=0.3, type=float,
                    help='Weight for medium memory in CMS (session patterns)')
parser.add_argument('--cms_slow_weight', default=0.2, type=float,
                    help='Weight for slow memory in CMS (long-term knowledge)')

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

    if args.use_nested_learning:
        print(f"Nested Learning (CMS): ENABLED")
        print(f"  - Fast memory weight: {args.cms_fast_weight}")
        print(f"  - Medium memory weight: {args.cms_medium_weight}")
        print(f"  - Slow memory weight: {args.cms_slow_weight}")

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
    
    # Apply Continuum Memory System if enabled
    if args.use_nested_learning:
        print("\n[Applying Continuum Memory System...]")
        original_emb = model.item_emb
        
        cms_emb = ContinuumItemEmbedding(
            num_items=original_emb.num_embeddings,
            embedding_dim=original_emb.embedding_dim,
            padding_idx=original_emb.padding_idx,
            fast_weight=args.cms_fast_weight,
            medium_weight=args.cms_medium_weight,
            slow_weight=args.cms_slow_weight,
            device=torch.device(args.device)
        )
        
        # Initialize CMS from current embeddings (random)
        print("  - Initializing CMS with random embeddings")
        with torch.no_grad():
            cms_emb.fast_emb.weight.copy_(original_emb.weight)
            cms_emb.medium_emb.weight.copy_(original_emb.weight)
            cms_emb.slow_emb.weight.copy_(original_emb.weight)
        
        # Replace item embeddings with CMS
        model.item_emb = cms_emb
        print("  ✓ CMS applied successfully")
    
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
    
    # Setup optimizer with multi-timescale learning rates for CMS
    if args.use_nested_learning:
        print("  - Using multi-timescale learning rates for CMS")
        
        # Get CMS parameter groups with different learning rates
        cms_param_groups = model.item_emb.get_parameter_groups(args.lr)
        
        # Get other model parameters
        other_params = [p for n, p in model.named_parameters() if 'item_emb' not in n]
        other_param_group = {'params': other_params, 'lr': args.lr, 'name': 'other'}
        
        # Combine all parameter groups
        param_groups = cms_param_groups + [other_param_group]
        
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), weight_decay=0.01)
        
        print(f"  - Fast memory LR: {args.lr:.6f}")
        print(f"  - Medium memory LR: {args.lr * 0.1:.6f}")
        print(f"  - Slow memory LR: {args.lr * 0.01:.6f}")
        print(f"  - Other params LR: {args.lr:.6f}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T, t0 = 0.0, time.time()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        epoch_loss, num_batches = 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", unit="batch", ncols=100)
        
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
            
            loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()

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