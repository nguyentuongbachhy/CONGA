import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from graph_model import (
    ItemCooccurrenceGraph,
    ItemGraphEmbedding,
    ItemGraphDataset,
    bpr_loss_item_graph,
    build_graph_from_dataset,
)


def str2bool(s: str) -> bool:
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser(description='Pretrain item graph embeddings')
parser.add_argument('--dataset', required=True, help='Dataset name (e.g., ml-1m)')
parser.add_argument('--output_dir', default='pretrained_embeddings', help='Output directory')
parser.add_argument('--output_name', default=None, type=str,
                    help='Output filename for embeddings (default: <dataset>_graph_embeddings.pt)')
parser.add_argument('--embedding_dim', default=50, type=int, help='Embedding dimension')
parser.add_argument('--num_layers', default=2, type=int, help='Number of GCN layers')
parser.add_argument('--window_size', default=10, type=int, help='Co-occurrence window size')
parser.add_argument('--min_cooccurrence', default=1, type=int, help='Min co-occurrence count')
parser.add_argument('--topk_sparsity', default=None, type=int, help='Top-k neighbors (None=all)')
parser.add_argument('--batch_size', default=2048, type=int, help='Batch size')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--num_epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--num_negatives', default=1, type=int, help='Number of negative samples')
parser.add_argument('--reg_weight', default=1e-4, type=float, help='L2 regularization weight')
parser.add_argument('--device', default='cuda', type=str, help='Device')
parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers')
parser.add_argument('--use_amp', default=True, type=str2bool, help='Use automatic mixed precision')
parser.add_argument('--eval_every', default=5, type=int, help='Evaluate every N epochs')

args = parser.parse_args()


def evaluate_graph_embeddings(model, adj_matrix, num_items, device, num_samples=1000):
    """
    Evaluate graph embeddings using link prediction.
    
    Args:
        model: ItemGraphEmbedding model
        adj_matrix: Adjacency matrix (sparse)
        num_items: Number of items
        device: Device
        num_samples: Number of test samples
        
    Returns:
        Tuple of (precision@10, recall@10)
    """
    model.eval()
    
    import numpy as np
    import scipy.sparse as sp
    
    adj_coo = adj_matrix.tocoo()
    edges = [(i, j) for i, j in zip(adj_coo.row, adj_coo.col) if i < j and i > 0 and j > 0]
    
    if len(edges) == 0:
        return 0.0, 0.0
    
    test_edges = np.random.choice(len(edges), min(num_samples, len(edges)), replace=False)
    
    precision_sum = 0.0
    recall_sum = 0.0
    
    with torch.no_grad():
        all_embeddings = model.get_all_embeddings()
        
        for edge_idx in test_edges:
            anchor, pos = edges[edge_idx]
            
            anchor_emb = all_embeddings[anchor].unsqueeze(0)
            scores = torch.mm(anchor_emb, all_embeddings.t()).squeeze(0)
            
            scores[anchor] = -float('inf')
            
            _, top_indices = torch.topk(scores, k=10)
            top_indices = top_indices.cpu().numpy()
            
            neighbors = adj_matrix[anchor].nonzero()[1]
            neighbors = set(neighbors) - {anchor}
            
            if len(neighbors) == 0:
                continue
            
            hits = len(set(top_indices) & neighbors)
            precision_sum += hits / 10.0
            recall_sum += hits / min(len(neighbors), 10)
    
    num_valid = len(test_edges)
    return precision_sum / num_valid, recall_sum / num_valid


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)

    output_name = args.output_name if args.output_name is not None else f'{args.dataset}_graph_embeddings.pt'
    output_path = os.path.join(args.output_dir, output_name)

    if os.name == 'nt' and args.num_workers != 0:
        print(f"\n[INFO] Windows detected: forcing DataLoader num_workers=0 (was {args.num_workers}) to avoid multiprocessing MemoryError")
        args.num_workers = 0
    
    print("=" * 60)
    print("PHASE 1: Graph Embedding Pre-training")
    print("=" * 60)
    
    print(f"\n[1/4] Building item co-occurrence graph...")
    adj_matrix, num_items = build_graph_from_dataset(
        args.dataset,
        window_size=args.window_size,
        min_cooccurrence=args.min_cooccurrence,
    )
    
    print(f"\n[2/4] Creating graph embedding model...")
    model = ItemGraphEmbedding(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        device=args.device,
        topk_sparsity=args.topk_sparsity,
    ).to(args.device)
    
    graph_builder = ItemCooccurrenceGraph(args.window_size, args.min_cooccurrence)
    norm_graph = graph_builder.normalize_adjacency(adj_matrix)
    model.set_graph(norm_graph)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n[3/4] Preparing training data...")
    train_dataset = ItemGraphDataset(
        adj_matrix=adj_matrix,
        num_items=num_items,
        num_negatives=args.num_negatives,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    use_amp = bool(args.use_amp) and str(args.device).startswith('cuda')
    scaler = torch.amp.grad_scaler.GradScaler(device='cuda') if use_amp else None
    
    print(f"\n[4/4] Training graph embeddings...")
    print(f"Config: {args.num_epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"AMP: {use_amp}, Device: {args.device}")
    print()
    
    best_precision = 0.0
    t0 = time.time()
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", unit="batch", ncols=100)
        
        for batch in pbar:
            if args.num_negatives == 1:
                anchor, pos, neg = batch
                anchor = anchor.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
            else:
                anchor, pos, neg = batch
                anchor = anchor.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast_mode.autocast(device_type='cuda', enabled=use_amp):
                anchor_emb, pos_emb, neg_emb = model(anchor, pos, neg)
                loss = bpr_loss_item_graph(
                    anchor_emb, pos_emb, neg_emb, reg_weight=args.reg_weight
                )
            
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
        elapsed = time.time() - t0
        
        print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s', end='')
        
        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            precision, recall = evaluate_graph_embeddings(
                model, adj_matrix, num_items, args.device
            )
            print(f' | P@10: {precision:.4f} | R@10: {recall:.4f}')
            
            if precision > best_precision:
                best_precision = precision
                model.save_embeddings(output_path)
                print(f'         ✓ Saved best embeddings (P@10={precision:.4f})')
        else:
            print()
    
    print("\n" + "=" * 60)
    print("Graph pre-training completed!")
    print(f"Best Precision@10: {best_precision:.4f}")
    print(f"Embeddings saved to: {output_path}")
    print("=" * 60)
