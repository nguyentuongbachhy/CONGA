import os
import argparse
import time
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

from models.graph_teacher import LightGCN, bpr_loss
from utils import (
    check_and_convert_dataset,
    load_metadata,
    data_partition,
    evaluate,
    evaluate_valid,
)


def str2bool(s: str) -> bool:
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


def sample_neg_items(
    user_train: Dict[int, List[int]], num_items: int, num_neg: int = 1
) -> Dict[int, List[int]]:
    user_neg = {}
    for user_id, items in user_train.items():
        if user_id == 0:
            continue
        pos_set = set(items)
        neg_items = []
        while len(neg_items) < num_neg:
            neg_item = np.random.randint(1, num_items + 1)
            if neg_item not in pos_set:
                neg_items.append(neg_item)
        user_neg[user_id] = neg_items
    return user_neg


def train_epoch(
    model: LightGCN,
    user_train: Dict[int, List[int]],
    num_items: int,
    batch_size: int,
    device: str,
    optimizer: torch.optim.Optimizer,
    num_neg: int = 1,
    epoch: int = 1,
    total_epochs: int = 100,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    users = list(user_train.keys())
    users = [u for u in users if u > 0 and len(user_train[u]) > 0]
    np.random.shuffle(users)

    pbar = tqdm(range(0, len(users), batch_size), desc=f"Epoch {epoch:3d}/{total_epochs}", unit="batch", ncols=100)
    
    for start_idx in pbar:
        batch_users = users[start_idx : start_idx + batch_size]

        batch_user_ids = []
        batch_pos_items = []
        batch_neg_items = []

        for user_id in batch_users:
            items = user_train[user_id]
            if len(items) == 0:
                continue

            pos_item = items[np.random.randint(len(items))]
            pos_set = set(items)

            neg_samples = []
            while len(neg_samples) < num_neg:
                neg_item = np.random.randint(1, num_items + 1)
                if neg_item not in pos_set:
                    neg_samples.append(neg_item)

            batch_user_ids.append(user_id)
            batch_pos_items.append(pos_item)
            batch_neg_items.append(neg_samples)

        if len(batch_user_ids) == 0:
            continue

        users_t = torch.LongTensor(batch_user_ids).to(device)
        pos_t = torch.LongTensor(batch_pos_items).to(device)
        neg_t = torch.LongTensor(batch_neg_items).to(device)

        user_emb, pos_emb, neg_emb = model(users_t, pos_t, neg_t)
        loss = bpr_loss(user_emb, pos_emb, neg_emb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})

    return total_loss / max(num_batches, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGCN teacher")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--maxlen", default=200, type=int)
    return parser.parse_args()


class Args:
    def __init__(self, maxlen: int, device: str) -> None:
        self.maxlen = maxlen
        self.device = device


def main() -> None:
    args = parse_args()

    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)

    print(f"Dataset: {args.dataset}")
    print(f"Users: {usernum}, Items: {itemnum}")

    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, _, _ = dataset

    os.makedirs(args.output_dir, exist_ok=True)

    model = LightGCN(
        num_users=usernum,
        num_items=itemnum,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        device=args.device,
    ).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Building graph...")
    model.build_graph(user_train)
    print("Graph built!")

    print("\n" + "=" * 70)
    print("Training LightGCN Teacher")
    print("=" * 70)
    print(f"Embedding dim: {args.embedding_dim} | Layers: {args.num_layers}")
    print(f"Batch size: {args.batch_size} | Epochs: {args.num_epochs} | LR: {args.lr}")
    print("=" * 70 + "\n")

    best_val_ndcg = 0.0
    best_test_ndcg = 0.0

    eval_args = Args(maxlen=args.maxlen, device=args.device)

    t0 = time.time()
    T = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        loss = train_epoch(
            model,
            user_train,
            itemnum,
            args.batch_size,
            args.device,
            optimizer,
            args.num_neg,
            epoch,
            args.num_epochs,
        )
        print(f"Epoch {epoch:3d} | Avg Loss: {loss:.4f}", end="")

        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            
            with torch.no_grad():
                t_valid = evaluate_valid(model, dataset, eval_args)
                t_test = evaluate(model, dataset, eval_args)

            print(f" | Time: {T:.1f}s")
            print(f"         Valid - NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}")
            print(f"         Test  - NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f}")

            if t_valid[0] > best_val_ndcg:
                best_val_ndcg = t_valid[0]
                best_test_ndcg = t_test[0]

                fname = f"lightgcn_teacher.dim={args.embedding_dim}.layers={args.num_layers}.pth"
                torch.save(model.state_dict(), os.path.join(args.output_dir, fname))
                print(f"         ✓ Best model saved: {fname}")
                print(f"         Best so far - Valid NDCG: {best_val_ndcg:.4f}, Test NDCG: {best_test_ndcg:.4f}")
            print()
            t0 = time.time()
        else:
            print()

    print("\nTraining completed!")
    print(f"Best Valid NDCG@10: {best_val_ndcg:.4f}")
    print(f"Best Test NDCG@10: {best_test_ndcg:.4f}")


if __name__ == "__main__":
    main()
