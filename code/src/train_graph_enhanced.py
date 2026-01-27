import os
import time
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.model import SASRec
from models.surge_integration import create_graph_enhanced_sasrec
from models.continuum_memory import ContinuumItemEmbedding
from utils import check_and_convert_dataset, load_metadata, get_dataloader, data_partition, evaluate, evaluate_valid
from losses import (
    listmle_loss, p_listmle_loss, p_sampled_softmax_loss, mmcl_loss, 
    gbce_loss, p_gbce_loss, rc_gbce_loss, approx_ndcg_loss, 
    infonce_gbce_loss, tcr_loss, composite_loss
)


def load_kgln_embeddings(checkpoint_path: str, expected_items: int, expected_dim: int):
    """Load pre-trained KGLN embeddings"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"KGLN checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    item_embeddings = checkpoint['item_embeddings']
    
    # Validate dimensions
    num_items = checkpoint['num_items']
    embedding_dim = checkpoint['embedding_dim']
    
    if num_items != expected_items:
        raise ValueError(f"Item count mismatch: expected {expected_items}, got {num_items}")
    
    if embedding_dim != expected_dim:
        raise ValueError(f"Embedding dim mismatch: expected {expected_dim}, got {embedding_dim}")
    
    print(f"✓ Loaded KGLN embeddings: {item_embeddings.shape}")
    print(f"  Pretrained epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Pretrained loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return item_embeddings


def main(args):
    print("="*80)
    print("CONGA with Graph Enhancement (KGLN + SURGE)")
    print("="*80)
    
    # Setup
    os.makedirs(args.dataset + "_" + args.train_dir, exist_ok=True)
    
    with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f_args:
        f_args.write("\n".join([f"{k},{v}" for k, v in sorted(vars(args).items())]))
    
    # Load dataset
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    
    print(f"\nDataset: {args.dataset}")
    print(f"Users: {usernum} | Items: {itemnum}")
    print(f"Loss: {args.loss_type} | Neg sampling: {args.neg_sampling_mode}")
    
    # Memory estimation
    tensor_mem_gb = (args.batch_size * args.maxlen * args.num_negatives * args.hidden_units * 4) / (1024**3)
    print(f"Estimated VRAM: {tensor_mem_gb:.2f} GB")
    
    if tensor_mem_gb > 4.0:
        print("[WARNING] High VRAM for 4GB GPU! Consider reducing batch_size or num_negatives")
    
    # Data
    train_loader = get_dataloader(
        args.dataset, args.maxlen, args.batch_size, mode="train",
        num_workers=args.num_workers, num_negatives=args.num_negatives,
        neg_sampling_mode=args.neg_sampling_mode
    )
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, _, _] = dataset
    
    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")
    
    # Create base SASRec model
    print("\n" + "="*80)
    print("Creating SASRec backbone...")
    print("="*80)
    
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    # Xavier initialization
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.item_emb.weight.data[0, :] = 0
    
    # Apply STEM if requested
    if args.use_stem:
        stem_layers = sorted(list(model.stem_layers))
        total_params = sum(p.numel() for p in model.parameters())
        stem_ratio = len(stem_layers) / args.num_blocks * 100
        print(f"\n{'='*60}")
        print(f"STEM Architecture")
        print(f"{'='*60}")
        print(f"STEM Layers: {stem_layers} ({stem_ratio:.0f}% of {args.num_blocks} blocks)")
        print(f"CPU Offload: {'Enabled' if args.stem_cpu_offload else 'Disabled'}")
        print(f"Total Params: {total_params:,}")
        print(f"{'='*60}\n")
    
    # Load KGLN embeddings and apply SURGE integration
    if args.kgln_checkpoint:
        print("\n" + "="*80)
        print("Loading KGLN Graph Embeddings...")
        print("="*80)
        
        graph_embeddings = load_kgln_embeddings(
            args.kgln_checkpoint,
            expected_items=itemnum,
            expected_dim=args.hidden_units,
        )
        
        print(f"\nApplying SURGE Integration...")
        print(f"  Injection mode: {args.injection_mode}")
        print(f"  Initial scale: {args.initial_scale}")
        print(f"  Final scale: {args.final_scale}")
        print(f"  Adaptive scaling: {args.use_adaptive_scale}")
        
        model = create_graph_enhanced_sasrec(
            model,
            graph_embeddings,
            injection_mode=args.injection_mode,
            use_adaptive_scale=args.use_adaptive_scale,
            initial_scale=args.initial_scale,
            final_scale=args.final_scale,
        ).to(args.device)
        
        print("✓ SURGE integration complete")
    else:
        print("\n[INFO] No KGLN checkpoint provided - training vanilla SASRec")
    
    # Apply CMS (Continuum Memory System) if requested
    if args.use_nested_learning:
        print("\n" + "="*80)
        print("Applying Continuum Memory System (CMS)...")
        print("="*80)
        
        # Access the actual item_emb from the model
        if args.kgln_checkpoint:
            # For SURGE model, get from sasrec submodule
            original_emb = getattr(model.sasrec, "item_emb")
        else:
            original_emb = getattr(model, "item_emb")
        
        cms_emb = ContinuumItemEmbedding(
            num_items=int(original_emb.num_embeddings),
            embedding_dim=int(original_emb.embedding_dim),
            padding_idx=int(original_emb.padding_idx) if original_emb.padding_idx is not None else 0,
            fast_weight=args.cms_fast_weight,
            medium_weight=args.cms_medium_weight,
            slow_weight=args.cms_slow_weight,
            device=torch.device(args.device),
        )
        
        with torch.no_grad():
            cms_emb.fast_emb.weight.copy_(original_emb.weight)
            cms_emb.medium_emb.weight.copy_(original_emb.weight)
            cms_emb.slow_emb.weight.copy_(original_emb.weight)
        
        if args.kgln_checkpoint:
            setattr(model.sasrec, "item_emb", cms_emb)
        else:
            setattr(model, "item_emb", cms_emb)
        
        print(f"✓ CMS applied | Weights: fast={args.cms_fast_weight}, "
              f"med={args.cms_medium_weight}, slow={args.cms_slow_weight}")
    
    # Load checkpoint if provided
    epoch_start_idx = 1
    if args.state_dict_path:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6:]
            epoch_start_idx = int(tail[:tail.find(".")]) + 1
            print(f"✓ Loaded checkpoint from epoch {epoch_start_idx - 1}")
        except Exception as e:
            print(f"[WARNING] Failed loading checkpoint: {e}")
    
    # Inference only mode
    if args.inference_only:
        print("\n[Inference Only Mode]")
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f"Test Results - NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f}")
        exit(0)
    
    # Setup optimizer
    if args.use_nested_learning:
        if args.kgln_checkpoint:
            item_emb = getattr(model.sasrec, "item_emb")
        else:
            item_emb = getattr(model, "item_emb")
        
        cms_param_groups = item_emb.get_parameter_groups(args.lr)
        other_params = [p for n, p in model.named_parameters() if "item_emb" not in n]
        param_groups = cms_param_groups + [{"params": other_params, "lr": args.lr, "name": "other"}]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), weight_decay=0.01)
        print(f"\nMulti-timescale LR: fast={args.lr:.6f}, med={args.lr * 0.1:.6f}, slow={args.lr * 0.01:.6f}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    
    # Training loop
    print("\n" + "="*80)
    print(f"Training for {args.num_epochs} epochs...")
    print("="*80 + "\n")
    
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T, t0 = 0.0, time.time()
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
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
            
            # Loss computation (same as original)
            if args.loss_type == "sampled_softmax":
                loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()
            elif args.loss_type == "p_sampled_softmax":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = p_sampled_softmax_loss(cand_logits, labels)
            elif args.loss_type == "gbce":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = gbce_loss(cand_logits, labels, alpha=alpha)
            elif args.loss_type == "composite":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss_weights = {"gbce": 0.5, "ndcg": 0.3, "infonce": 0.2}
                loss = composite_loss(cand_logits, labels, loss_weights=loss_weights, k=10)
            else:
                # Default to sampled_softmax for other losses
                loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}", end="")
        
        # Evaluation
        if epoch % 20 == 0:
            model.eval()
            T += time.time() - t0
            with torch.no_grad():
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
            
            print(f" | Time: {T:.1f}s")
            print(f"         Valid: {t_valid} | Test: {t_test}")
            
            if t_valid[0] > best_val_ndcg:
                best_val_ndcg, best_val_hr = t_valid
                best_test_ndcg, best_test_hr = t_test
                torch.save(model.state_dict(), os.path.join(args.dataset + "_" + args.train_dir, "SASRec.best.pth"))
                print("         ✓ Saved best model")
            
            f.write(f"{epoch} {t_valid} {t_test}\n")
            f.flush()
            t0 = time.time()
        else:
            print()
    
    f.close()
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best Val  - NDCG@10: {best_val_ndcg:.4f}, HR@10: {best_val_hr:.4f}")
    print(f"Best Test - NDCG@10: {best_test_ndcg:.4f}, HR@10: {best_test_hr:.4f}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CONGA with Graph Enhancement")
    
    # Dataset
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--train_dir", required=True, help="Training directory")
    
    # Model architecture
    parser.add_argument("--maxlen", default=200, type=int, help="Max sequence length")
    parser.add_argument("--hidden_units", default=50, type=int, help="Hidden dimension")
    parser.add_argument("--num_blocks", default=2, type=int, help="Number of transformer blocks")
    parser.add_argument("--num_heads", default=1, type=int, help="Number of attention heads")
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="Dropout rate")
    
    # Training
    parser.add_argument("--num_epochs", default=1000, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--num_negatives", default=1, type=int, help="Number of negatives")
    parser.add_argument("--neg_sampling_mode", default="random", type=str, 
                        choices=["random", "popularity", "frequency", "mans"])
    parser.add_argument("--loss_type", default="sampled_softmax", type=str,
                        choices=["sampled_softmax", "p_sampled_softmax", "gbce", "composite"])
    
    # Graph enhancement (KGLN + SURGE)
    parser.add_argument("--kgln_checkpoint", type=str, default=None, 
                        help="Path to pre-trained KGLN checkpoint")
    parser.add_argument("--injection_mode", type=str, default="gate",
                        choices=["gate", "concat", "residual"],
                        help="SURGE injection mode")
    parser.add_argument("--use_adaptive_scale", action="store_true", default=True,
                        help="Use adaptive scaling (0.2 -> 0.4)")
    parser.add_argument("--initial_scale", type=float, default=0.2,
                        help="Initial graph embedding scale")
    parser.add_argument("--final_scale", type=float, default=0.4,
                        help="Final graph embedding scale")
    
    # STEM
    parser.add_argument("--use_stem", default=False, action="store_true")
    parser.add_argument("--stem_layers", default=None, type=str)
    parser.add_argument("--stem_cpu_offload", default=False, action="store_true")
    
    # Continuum Memory System (CMS)
    parser.add_argument("--use_nested_learning", default=False, action="store_true")
    parser.add_argument("--cms_fast_weight", default=0.9, type=float)
    parser.add_argument("--cms_medium_weight", default=0.05, type=float)
    parser.add_argument("--cms_slow_weight", default=0.05, type=float)
    
    # System
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--norm_first", action="store_true", default=False)
    parser.add_argument("--inference_only", default=False, action="store_true")
    parser.add_argument("--state_dict_path", default=None, type=str)
    
    args = parser.parse_args()
    
    # Parse stem_layers
    if args.stem_layers is not None:
        args.stem_layers = [int(x.strip()) for x in args.stem_layers.split(',')]
    
    main(args)
