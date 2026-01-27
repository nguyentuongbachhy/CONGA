"""
Train SASRec with KGLN pretrained embeddings
Usage:
    python train_with_kgln.py --dataset ml-1m --kgln_path pretrained_embeddings/kgln_best.pth
"""

import os
import time
import torch
import argparse
import torch.nn.functional as F
import multiprocessing
from tqdm import tqdm

multiprocessing.set_start_method('spawn', force=True)
from models.model import SASRec
from utils import load_metadata, data_partition, evaluate, evaluate_valid
from losses import gbce_loss


def load_kgln_embeddings(kgln_path, expected_num_items=None):
    """Load KGLN pretrained embeddings"""
    print(f"\nLoading KGLN embeddings from: {kgln_path}")
    
    checkpoint = torch.load(kgln_path, map_location='cpu')
    
    item_embeddings = checkpoint['item_embeddings']
    num_items = checkpoint['num_items']
    embedding_dim = checkpoint['embedding_dim']
    
    print(f"  Items: {num_items:,}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Training loss: {checkpoint.get('loss', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    if expected_num_items is not None and num_items != expected_num_items:
        print(f"  WARNING: Expected {expected_num_items} items, got {num_items}")
    
    return item_embeddings, embedding_dim


def initialize_sasrec_with_kgln(
    sasrec_model,
    kgln_embeddings,
    fusion_mode='add',
    fusion_weight=0.3,
):
    """Initialize SASRec item embeddings with KGLN embeddings"""
    
    print(f"\nInitializing SASRec with KGLN embeddings:")
    print(f"  Fusion mode: {fusion_mode}")
    print(f"  Fusion weight: {fusion_weight}")
    
    # Get SASRec item embeddings
    sasrec_item_emb = sasrec_model.item_emb.weight.data
    
    # Move KGLN embeddings to same device as model
    kgln_embeddings = kgln_embeddings.to(sasrec_item_emb.device)
    
    # Ensure dimensions match
    if sasrec_item_emb.size(1) != kgln_embeddings.size(1):
        raise ValueError(
            f"Dimension mismatch: SASRec={sasrec_item_emb.size(1)}, "
            f"KGLN={kgln_embeddings.size(1)}"
        )
    
    # Ensure number of items match
    num_items = min(sasrec_item_emb.size(0), kgln_embeddings.size(0))
    
    if fusion_mode == 'replace':
        # Replace SASRec embeddings with KGLN embeddings
        sasrec_model.item_emb.weight.data[:num_items] = kgln_embeddings[:num_items]
        print(f"  [OK] Replaced {num_items} item embeddings")
    
    elif fusion_mode == 'add':
        # Add weighted KGLN embeddings to SASRec embeddings
        sasrec_model.item_emb.weight.data[:num_items] += (
            fusion_weight * kgln_embeddings[:num_items]
        )
        print(f"  [OK] Added weighted KGLN embeddings to {num_items} items")
    
    elif fusion_mode == 'interpolate':
        # Interpolate between SASRec and KGLN embeddings
        sasrec_model.item_emb.weight.data[:num_items] = (
            (1 - fusion_weight) * sasrec_item_emb[:num_items] +
            fusion_weight * kgln_embeddings[:num_items]
        )
        print(f"  [OK] Interpolated {num_items} item embeddings")
    
    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    
    return sasrec_model


def main():
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--train_dir', default='kgln_training', type=str)
    
    # KGLN embeddings
    parser.add_argument('--kgln_path', required=True, type=str,
                        help='Path to KGLN pretrained embeddings (kgln_best.pth)')
    parser.add_argument('--fusion_mode', default='add', type=str,
                        choices=['replace', 'add', 'interpolate'],
                        help='How to fuse KGLN embeddings with SASRec')
    parser.add_argument('--fusion_weight', default=0.3, type=float,
                        help='Weight for KGLN embeddings (0-1)')
    
    # Model architecture
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--maxlen', default=512, type=int)
    parser.add_argument('--norm_first', action='store_true', help='Use pre-norm instead of post-norm')
    
    # Training
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs', default=600, type=int)
    parser.add_argument('--num_negatives', default=16, type=int)
    parser.add_argument('--loss_type', default='gbce', type=str, 
                        choices=['sampled_softmax', 'gbce'],
                        help='Loss function type')
    
    # System
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.train_dir, exist_ok=True)
    
    print("="*80)
    print("TRAIN SASREC WITH KGLN EMBEDDINGS")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"KGLN path: {args.kgln_path}")
    print(f"Fusion mode: {args.fusion_mode}")
    print(f"Fusion weight: {args.fusion_weight}")
    print(f"Hidden units: {args.hidden_units}")
    print(f"Num blocks: {args.num_blocks}")
    print(f"Max length: {args.maxlen}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Negatives: {args.num_negatives}")
    print(f"Loss type: {args.loss_type}")
    print(f"Norm first: {args.norm_first}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    
    usernum, itemnum = load_metadata(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    print(f"Users: {usernum:,}")
    print(f"Items: {itemnum:,}")
    print(f"Train users: {len(user_train):,}")
    
    # Load KGLN embeddings
    kgln_embeddings, kgln_dim = load_kgln_embeddings(
        args.kgln_path,
        expected_num_items=itemnum
    )
    
    # Check dimension compatibility
    if args.hidden_units != kgln_dim:
        print(f"\nWARNING: SASRec hidden_units ({args.hidden_units}) != "
              f"KGLN embedding_dim ({kgln_dim})")
        print(f"Setting hidden_units = {kgln_dim} to match KGLN embeddings")
        args.hidden_units = kgln_dim
    
    # Create dataloader
    from utils.common import SASRecDataset
    from torch.utils.data import DataLoader
    
    train_dataset = SASRecDataset(
        dataset_name=args.dataset,
        maxlen=args.maxlen,
        mode='train',
        num_negatives=args.num_negatives,
        neg_sampling_mode='random',
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid Windows multiprocessing issues
        pin_memory=True,
    )
    
    # Initialize SASRec model
    print("\nInitializing SASRec model...")
    model = SASRec(
        user_num=usernum,
        item_num=itemnum,
        args=args,
    ).to(args.device)
    
    # Initialize with KGLN embeddings
    model = initialize_sasrec_with_kgln(
        model,
        kgln_embeddings,
        fusion_mode=args.fusion_mode,
        fusion_weight=args.fusion_weight,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T, t0 = 0.0, time.time()
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss, num_batches = 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", unit="batch", ncols=100)
        
        for step, batch in enumerate(pbar):
            u, seq, pos, neg = [x.to(args.device) for x in batch]
            optimizer.zero_grad()
            
            mask = pos != 0
            mask_exp = mask.unsqueeze(-1).expand(-1, -1, args.num_negatives)
            
            pos_logits, neg_logits = model(u, seq, pos, neg)
            
            pos_sel = torch.masked_select(pos_logits, mask)
            neg_sel = torch.masked_select(neg_logits, mask).unsqueeze(1) if neg_logits.dim() == 2 else torch.masked_select(neg_logits, mask_exp).view(-1, args.num_negatives)
            
            cand_logits = torch.cat([pos_sel.unsqueeze(1), neg_sel], dim=1)
            
            if args.loss_type == "sampled_softmax":
                loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()
            elif args.loss_type == "gbce":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = gbce_loss(cand_logits, labels)
            else:
                raise ValueError(f"Unknown loss_type: {args.loss_type}")
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluate every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            
            print(f"\nEpoch {epoch:3d} | Loss: {avg_loss:.4f} | Time: {t1:.1f}s")
            print("Evaluating...")
            
            t_valid = evaluate_valid(model, dataset, args)
            t_test = evaluate(model, dataset, args)
            
            val_ndcg, val_hr = t_valid[0], t_valid[1]
            test_ndcg, test_hr = t_test[0], t_test[1]
            
            print(f"Valid: NDCG@10={val_ndcg:.4f}, HR@10={val_hr:.4f}")
            print(f"Test:  NDCG@10={test_ndcg:.4f}, HR@10={test_hr:.4f}")
            
            # Save best model
            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_val_hr = val_hr
                best_test_ndcg = test_ndcg
                best_test_hr = test_hr
                
                save_path = os.path.join(args.train_dir, f'best_model_epoch={epoch}.pth')
                torch.save(model.state_dict(), save_path)
                print(f"[OK] New best model saved! (Val NDCG@10: {val_ndcg:.4f})")
            
            t0 = time.time()
        else:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")
    
    # Final results
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Validation: NDCG@10={best_val_ndcg:.4f}, HR@10={best_val_hr:.4f}")
    print(f"Best Test:       NDCG@10={best_test_ndcg:.4f}, HR@10={best_test_hr:.4f}")
    print(f"Total time: {T/3600:.2f} hours")
    print("="*80)


if __name__ == '__main__':
    main()
