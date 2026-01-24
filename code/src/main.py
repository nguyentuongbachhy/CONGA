import os
import time
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from models.model import SASRec
from models.continuum_memory import ContinuumItemEmbedding
from utils import check_and_convert_dataset, load_metadata, get_dataloader, data_partition, evaluate, evaluate_valid
from losses import listmle_loss, p_listmle_loss, p_sampled_softmax_loss, mmcl_loss, gbce_loss, p_gbce_loss, rc_gbce_loss, approx_ndcg_loss, infonce_gbce_loss, tcr_loss, composite_loss


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=200, type=int)
parser.add_argument("--hidden_units", default=50, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--num_negatives", default=1, type=int)
parser.add_argument("--neg_sampling_mode", default="random", type=str, choices=["random", "popularity", "frequency", "mans"])
parser.add_argument("--loss_type", default="sampled_softmax", type=str, choices=["sampled_softmax", "p_sampled_softmax", "listmle", "p_listmle", "mmcl", "gbce", "p_gbce", "rc_gbce", "approx_ndcg", "infonce_gbce", "tcr", "composite"])
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--inference_only", default=False, action="store_true")
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument("--norm_first", action="store_true", default=False)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--use_nested_learning", default=False, action="store_true")
parser.add_argument("--cms_fast_weight", default=0.5, type=float)
parser.add_argument("--cms_medium_weight", default=0.3, type=float)
parser.add_argument("--cms_slow_weight", default=0.2, type=float)

args = parser.parse_args()


if __name__ == "__main__":
    os.makedirs(args.dataset + "_" + args.train_dir, exist_ok=True)
    
    with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f_args:
        f_args.write("\n".join([f"{k},{v}" for k, v in sorted(vars(args).items())]))
    
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    
    tensor_mem_gb = (args.batch_size * args.maxlen * args.num_negatives * args.hidden_units * 4) / (1024**3)
    print(f"Dataset: {args.dataset} | Users: {usernum} | Items: {itemnum}")
    print(f"Estimated VRAM: {tensor_mem_gb:.2f} GB")
    
    if tensor_mem_gb > 8.0:
        print("[WARNING] High VRAM required! Reduce --num_negatives or --batch_size")
    
    print(f"Loss: {args.loss_type} | Neg sampling: {args.neg_sampling_mode}")
    
    train_loader = get_dataloader(args.dataset, args.maxlen, args.batch_size, mode="train", num_workers=args.num_workers, num_negatives=args.num_negatives, neg_sampling_mode=args.neg_sampling_mode)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, _, _] = dataset
    
    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")
    
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.item_emb.weight.data[0, :] = 0
    
    if args.use_nested_learning:
        print("Applying CMS...")
        original_emb: torch.nn.Embedding = getattr(model, "item_emb")
        
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
        
        setattr(model, "item_emb", cms_emb)
        print("✓ CMS applied")
    
    epoch_start_idx = 1
    if args.state_dict_path:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6:]
            epoch_start_idx = int(tail[:tail.find(".")]) + 1
            print(f"Loaded checkpoint from epoch {epoch_start_idx - 1}")
        except Exception as e:
            print(f"Failed loading checkpoint: {e}")
    
    if args.inference_only:
        print("[Inference Only]")
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f"Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})")
        exit(0)
    
    if args.use_nested_learning:
        item_emb = getattr(model, "item_emb")
        cms_param_groups = item_emb.get_parameter_groups(args.lr)
        other_params = [p for n, p in model.named_parameters() if "item_emb" not in n]
        param_groups = cms_param_groups + [{"params": other_params, "lr": args.lr, "name": "other"}]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), weight_decay=0.01)
        print(f"Multi-timescale LR: fast={args.lr:.6f}, med={args.lr * 0.1:.6f}, slow={args.lr * 0.01:.6f}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    
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
            
            if args.loss_type == "sampled_softmax":
                loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()
            elif args.loss_type == "p_sampled_softmax":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = p_sampled_softmax_loss(cand_logits, labels)
            elif args.loss_type == "listmle":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = listmle_loss(cand_logits, labels)
            elif args.loss_type == "p_listmle":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = p_listmle_loss(cand_logits, labels)
            elif args.loss_type == "mmcl":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                if args.num_negatives <= 3:
                    margins, weights = [0.3, 0.6], [1.0, 0.5]
                elif args.num_negatives <= 10:
                    margins, weights = [0.2, 0.5, 0.8], [1.0, 0.5, 0.2]
                else:
                    margins, weights = [0.1, 0.3, 0.6, 0.9], [1.0, 0.7, 0.4, 0.2]
                loss = mmcl_loss(cand_logits, labels, margins=margins, weights=weights)
            elif args.loss_type == "gbce":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = gbce_loss(cand_logits, labels, alpha=alpha)
            elif args.loss_type == "p_gbce":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = p_gbce_loss(cand_logits, labels, alpha=alpha)
            elif args.loss_type == "rc_gbce":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = rc_gbce_loss(cand_logits, labels, alpha=alpha, k=10)
            elif args.loss_type == "approx_ndcg":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss = approx_ndcg_loss(cand_logits, labels, k=10)
            elif args.loss_type == "infonce_gbce":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = infonce_gbce_loss(cand_logits, labels, alpha=alpha, beta=0.3)
            elif args.loss_type == "tcr":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                alpha = 0.75 if args.num_negatives >= 5 else 0.7
                loss = tcr_loss(cand_logits, labels, alpha=alpha)
            elif args.loss_type == "composite":
                labels = torch.zeros_like(cand_logits)
                labels[:, 0] = 1.0
                loss_weights = {"gbce": 0.5, "ndcg": 0.3, "infonce": 0.2}
                loss = composite_loss(cand_logits, labels, loss_weights=loss_weights, k=10)
            else:
                raise ValueError(f"Unknown loss_type: {args.loss_type}")

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}", end="")
        
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
    print("\nTraining completed!")
    print(f"Best Val  - NDCG@10: {best_val_ndcg:.4f}, HR@10: {best_val_hr:.4f}")
    print(f"Best Test - NDCG@10: {best_test_ndcg:.4f}, HR@10: {best_test_hr:.4f}")
