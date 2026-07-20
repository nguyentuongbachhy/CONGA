import os
import time
import argparse
import torch
import torch.nn.functional as F
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

from model import SASRec
from utils import check_and_convert_dataset, load_metadata, get_dataloader, data_partition, evaluate, evaluate_valid
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--num_negatives', default=1, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--grad_clip', default=0.0, type=float)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--inference_only', default=False, action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision('high')

    os.makedirs(args.train_dir, exist_ok=True)
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f_args:
        f_args.write('\n'.join(f'{k},{v}' for k, v in sorted(vars(args).items())))

    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    print(f'Dataset: {args.dataset} | Users: {usernum} | Items: {itemnum}')

    train_loader = get_dataloader(
        args.dataset, args.maxlen, args.batch_size, mode='train',
        num_workers=args.num_workers, num_negatives=1,  # neg sampled on-the-fly
    )
    dataset = data_partition(args.dataset)

    f = open(os.path.join(args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg5, val_hr5, val_ndcg10, val_hr10) (test_ndcg5, test_hr5, test_ndcg10, test_hr10)\n')

    model = SASRec(usernum, itemnum, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.item_emb.weight.data[0, :] = 0.0

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f'Loaded checkpoint from epoch {epoch_start_idx - 1}')
        except Exception as e:
            print(f'Failed loading checkpoint: {e}')

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f'Test NDCG@10: {t_test[2]:.4f}  HR@10: {t_test[3]:.4f}')
        exit(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)

    out_dir = args.train_dir
    best_ckpt_path = os.path.join(out_dir, 'best_model.pth')
    best_val_ndcg10 = 0.0
    T, t0 = 0.0, time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        epoch_loss, num_batches = 0.0, 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:3d}', unit='batch', ncols=100)
        for batch in pbar:
            batch_tensors = [x.to(args.device, non_blocking=True) for x in batch]
            u, seq, pos = batch_tensors[0], batch_tensors[1], batch_tensors[2]

            with torch.no_grad():
                neg = torch.randint(
                    1, itemnum + 1,
                    (u.size(0), args.maxlen, args.num_negatives),
                    dtype=torch.long, device=args.device,
                )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                pos_logits, neg_logits, _, _ = model(u, seq, pos, neg)

                mask = pos != 0
                pos_sel = pos_logits[mask]
                if neg_logits.dim() == 3:
                    neg_sel = neg_logits[mask]
                else:
                    neg_sel = neg_logits[mask].unsqueeze(1)

                loss = F.binary_cross_entropy_with_logits(
                    pos_sel, torch.ones_like(pos_sel)
                ) + F.binary_cross_entropy_with_logits(
                    neg_sel, torch.zeros_like(neg_sel)
                )

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f'{epoch_loss / num_batches:.4f}')

        if epoch % 10 == 0:
            t1 = time.time() - t0
            T += t1
            model.eval()
            with torch.no_grad():
                t_valid = evaluate_valid(model, dataset, args)
                t_test = evaluate(model, dataset, args)

            is_best = t_valid[2] > best_val_ndcg10
            if is_best:
                best_val_ndcg10 = t_valid[2]
                torch.save(model.state_dict(), best_ckpt_path)

            print(
                f'epoch:{epoch:3d}  time:{T:.1f}s  '
                f'val(NDCG@5:{t_valid[0]:.4f} HR@5:{t_valid[1]:.4f} NDCG@10:{t_valid[2]:.4f} HR@10:{t_valid[3]:.4f})  '
                f'test(NDCG@5:{t_test[0]:.4f} HR@5:{t_test[1]:.4f} NDCG@10:{t_test[2]:.4f} HR@10:{t_test[3]:.4f})'
                + ('  [best]' if is_best else '')
            )
            f.write(f'{epoch} {t_valid} {t_test}\n')
            f.flush()
            t0 = time.time()

    f.close()
    print(f'Done. Best model saved to {best_ckpt_path} (val NDCG@10={best_val_ndcg10:.4f})')

