import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import SASRec
from sasrec_integration import (
    create_sasrec_with_graph_init,
    initialize_sasrec_with_graph_embeddings,
)
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


parser = argparse.ArgumentParser(description='Train SASRec with graph-initialized embeddings')
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--graph_embedding_path', default=None, type=str, 
                    help='Path to pretrained graph embeddings')
parser.add_argument('--freeze_embeddings', default=False, type=str2bool,
                    help='Freeze item embeddings during training')
parser.add_argument('--graph_scale_factor', default=1.0, type=float,
                    help='Scale factor for graph embeddings (0.0-1.0, lower = less influence)')
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
parser.add_argument('--num_workers', default=4, type=int)

args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f_args:
        f_args.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    
    tensor_mem_gb = (args.batch_size * args.maxlen * args.num_negatives * args.hidden_units * 4) / (1024**3)
    print("=" * 60)
    print("PHASE 2: SASRec Fine-tuning with Graph Initialization")
    print("=" * 60)
    print(f"Dataset: {args.dataset} | Users: {usernum} | Items: {itemnum}")
    print(f"Estimated VRAM for Negatives Tensor: {tensor_mem_gb:.2f} GB")
    
    if tensor_mem_gb > 8.0:
        print("\n[WARNING] Cấu hình này yêu cầu VRAM rất lớn!")
        print("Gợi ý: Giảm --num_negatives (ví dụ: 1) hoặc giảm --batch_size.\n")
    
    if args.graph_embedding_path:
        print(f"Graph embeddings: {args.graph_embedding_path}")
        print(f"Freeze embeddings: {args.freeze_embeddings}")
    else:
        print("No graph initialization (training from scratch)")
    
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
    
    print("\n[1/3] Creating SASRec model...")
    model = create_sasrec_with_graph_init(
        usernum,
        itemnum,
        args,
        graph_embedding_path=args.graph_embedding_path,
        freeze_embeddings=args.freeze_embeddings,
        scale_factor=args.graph_scale_factor,
    )
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f"Loaded checkpoint from epoch {epoch_start_idx - 1}")
        except Exception as e:
            print(f'Failed loading state_dicts: {e}')
    
    if args.inference_only:
        print("\n[Inference Only Mode]")
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
        exit(0)
    
    print("\n[2/3] Setting up optimizer...")
    if args.freeze_embeddings:
        trainable_params = [p for n, p in model.named_parameters() if 'item_emb' not in n]
        print(f"Training {len(trainable_params)} parameter groups (embeddings frozen)")
    else:
        trainable_params = model.parameters()
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T, t0 = 0.0, time.time()
    
    print("\n[3/3] Training SASRec...")
    print()
    
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
    
    print("\n" + "=" * 60)
    print("SASRec training completed!")
    print(f"Best Validation - NDCG@10: {best_val_ndcg:.4f}, HR@10: {best_val_hr:.4f}")
    print(f"Best Test       - NDCG@10: {best_test_ndcg:.4f}, HR@10: {best_test_hr:.4f}")
    print("=" * 60)
