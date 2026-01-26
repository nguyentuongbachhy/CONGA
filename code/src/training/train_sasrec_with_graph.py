import os
import time
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.sasrec_integration import create_sasrec_with_graph_init
from models.continuum_memory import ContinuumItemEmbedding
from utils import check_and_convert_dataset, load_metadata, get_dataloader, data_partition, evaluate, evaluate_valid


def listmle_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    _, indices = y_true.sort(descending=True, dim=-1)
    preds_sorted_by_true = torch.gather(y_pred, dim=1, index=indices)
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    return torch.mean(torch.sum(observation_loss, dim=1))


def p_listmle_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    y_true_sorted, indices = y_true.sort(descending=True, dim=-1)
    preds_sorted_by_true = torch.gather(y_pred, dim=1, index=indices)
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    slate_length = y_pred.shape[1]
    positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=y_pred.device)
    position_weights = (1.0 / torch.log2(positions + 1.0)).unsqueeze(0)
    weighted_loss = observation_loss * position_weights
    return torch.mean(torch.sum(weighted_loss, dim=1))


def p_sampled_softmax_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    softmax_probs = F.log_softmax(y_pred, dim=1)
    base_loss = -softmax_probs[:, 0]
    slate_length = y_pred.shape[1]
    positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=y_pred.device)
    position_weights = (1.0 / torch.log2(positions + 1.0)).unsqueeze(0)
    weighted_loss = base_loss * position_weights[:, 0]
    return torch.mean(weighted_loss)


def mmcl_loss(y_pred: torch.Tensor, y_true: torch.Tensor, margins=[0.2, 0.5, 0.8], weights=[1.0, 0.5, 0.2], temperature=1.0, eps=1e-10) -> torch.Tensor:
    y_pred_scaled = y_pred / temperature
    pos_sim = y_pred_scaled[:, 0]
    neg_sims = y_pred_scaled[:, 1:]
    pos_loss = torch.mean(torch.relu(1.0 - pos_sim))
    neg_loss = 0.0
    for margin, weight in zip(margins, weights):
        margin_term = neg_sims - margin
        neg_loss += weight * torch.mean(F.softplus(margin_term))
    return pos_loss + neg_loss


def gbce_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha=0.75, temperature=1.0, eps=1e-10) -> torch.Tensor:
    y_pred_scaled = y_pred / temperature
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_loss = -(y_true * torch.log(sigmoid_pred + eps) + (1 - y_true) * torch.log(1 - sigmoid_pred + eps))
    bce_loss = torch.mean(bce_loss)
    ce_loss = -F.log_softmax(y_pred_scaled, dim=1)[:, 0]
    ce_loss = torch.mean(ce_loss)
    return alpha * bce_loss + (1 - alpha) * ce_loss


def p_gbce_loss(y_pred: torch.Tensor, y_true: torch.Tensor, alpha=0.75, temperature=1.0, eps=1e-10) -> torch.Tensor:
    y_pred_scaled = y_pred / temperature
    slate_length = y_pred.shape[1]
    positions = torch.arange(1, slate_length + 1, dtype=torch.float32, device=y_pred.device)
    position_weights = (1.0 / torch.log2(positions + 1.0)).unsqueeze(0)
    sigmoid_pred = torch.sigmoid(y_pred_scaled)
    bce_loss_per_item = -(y_true * torch.log(sigmoid_pred + eps) + (1 - y_true) * torch.log(1 - sigmoid_pred + eps))
    weighted_bce_loss = torch.mean(bce_loss_per_item * position_weights)
    ce_loss_per_item = -F.log_softmax(y_pred_scaled, dim=1)
    weighted_ce_loss = ce_loss_per_item * position_weights
    positive_mask = y_true == 1.0
    weighted_ce_loss = torch.mean(weighted_ce_loss * positive_mask)
    return alpha * weighted_bce_loss + (1 - alpha) * weighted_ce_loss


parser = argparse.ArgumentParser(description='Train SASRec with graph-initialized embeddings')
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--graph_embedding_path', default=None, type=str)
parser.add_argument('--freeze_embeddings', default=False, action='store_true')
parser.add_argument('--graph_scale_factor', default=1.0, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--num_negatives', default=1, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--use_nested_learning', default=False, action='store_true')
parser.add_argument('--cms_fast_weight', default=0.5, type=float)
parser.add_argument('--cms_medium_weight', default=0.3, type=float)
parser.add_argument('--cms_slow_weight', default=0.2, type=float)
parser.add_argument('--loss_type', default='sampled_softmax', type=str, choices=['sampled_softmax', 'p_sampled_softmax', 'listmle', 'p_listmle', 'mmcl', 'gbce', 'p_gbce'])

args = parser.parse_args()


if __name__ == '__main__':
    os.makedirs(args.dataset + '_' + args.train_dir, exist_ok=True)
    
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f_args:
        f_args.write('\n'.join([f'{k},{v}' for k, v in sorted(vars(args).items())]))
    
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    
    tensor_mem_gb = (args.batch_size * args.maxlen * args.num_negatives * args.hidden_units * 4) / (1024**3)
    print("=" * 60)
    print("PHASE 2: SASRec Fine-tuning with Graph Initialization")
    print("=" * 60)
    print(f"Dataset: {args.dataset} | Users: {usernum} | Items: {itemnum}")
    print(f"Estimated VRAM: {tensor_mem_gb:.2f} GB")
    
    if tensor_mem_gb > 8.0:
        print("\n[WARNING] High VRAM required!")
        print("Suggestion: Reduce --num_negatives or --batch_size\n")
    
    print(f"Loss: {args.loss_type}")
    if args.use_nested_learning:
        print(f"CMS: ENABLED (fast={args.cms_fast_weight}, med={args.cms_medium_weight}, slow={args.cms_slow_weight})")
    if args.graph_embedding_path:
        print(f"Graph: {args.graph_embedding_path} (freeze={args.freeze_embeddings})")
    
    train_loader = get_dataloader(args.dataset, args.maxlen, args.batch_size, mode='train', num_workers=args.num_workers, num_negatives=args.num_negatives)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, _, _] = dataset
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    print("\n[1/3] Creating SASRec model...")
    model = create_sasrec_with_graph_init(usernum, itemnum, args, graph_embedding_path=args.graph_embedding_path, freeze_embeddings=args.freeze_embeddings, scale_factor=args.graph_scale_factor)
    
    if args.use_nested_learning:
        print("\n[1.5/3] Applying CMS...")
        original_emb: torch.nn.Embedding = getattr(model, 'item_emb')
        
        cms_emb = ContinuumItemEmbedding(
            num_items=int(original_emb.num_embeddings),
            embedding_dim=int(original_emb.embedding_dim),
            padding_idx=int(original_emb.padding_idx) if original_emb.padding_idx is not None else 0,
            fast_weight=args.cms_fast_weight,
            medium_weight=args.cms_medium_weight,
            slow_weight=args.cms_slow_weight,
            device=torch.device(args.device)
        )
        
        if args.graph_embedding_path:
            print("  - Initializing CMS with graph embeddings")
            cms_emb.init_from_pretrained(original_emb.weight.data)
        else:
            print("  - Initializing CMS with random embeddings")
            with torch.no_grad():
                cms_emb.fast_emb.weight.copy_(original_emb.weight)
                cms_emb.medium_emb.weight.copy_(original_emb.weight)
                cms_emb.slow_emb.weight.copy_(original_emb.weight)
        
        setattr(model, 'item_emb', cms_emb)
        print("  Γ£ô CMS applied")
    
    epoch_start_idx = 1
    if args.state_dict_path:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f"Loaded checkpoint from epoch {epoch_start_idx - 1}")
        except Exception as e:
            print(f'Failed loading checkpoint: {e}')
    
    if args.inference_only:
        print("\n[Inference Only]")
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f'Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')
        exit(0)
    
    print("\n[2/3] Setting up optimizer...")
    
    if args.use_nested_learning and not args.freeze_embeddings:
        print("  - Multi-timescale LR for CMS")
        item_emb = getattr(model, 'item_emb')
        cms_param_groups = item_emb.get_parameter_groups(args.lr)
        other_params = [p for n, p in model.named_parameters() if 'item_emb' not in n]
        param_groups = cms_param_groups + [{'params': other_params, 'lr': args.lr, 'name': 'other'}]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), weight_decay=0.01)
        print(f"  - Fast: {args.lr:.6f}, Med: {args.lr * 0.1:.6f}, Slow: {args.lr * 0.01:.6f}")
    elif args.freeze_embeddings:
        trainable_params = [p for n, p in model.named_parameters() if 'item_emb' not in n]
        print(f"  - Embeddings frozen")
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T, t0 = 0.0, time.time()
    
    print("\n[3/3] Training...\n")
    
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
            neg_sel = torch.masked_select(neg_logits, mask).unsqueeze(1) if neg_logits.dim() == 2 else torch.masked_select(neg_logits, mask_exp).view(-1, args.num_negatives)
            
            cand_logits = torch.cat([pos_sel.unsqueeze(1), neg_sel], dim=1)
            
            if args.loss_type == 'sampled_softmax':
                loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()
            elif args.loss_type == 'p_sampled_softmax':
                cand_labels = torch.zeros_like(cand_logits)
                cand_labels[:, 0] = 1.0
                loss = p_sampled_softmax_loss(cand_logits, cand_labels)
            elif args.loss_type == 'listmle':
                cand_labels = torch.zeros_like(cand_logits)
                cand_labels[:, 0] = 1.0
                loss = listmle_loss(cand_logits, cand_labels)
            elif args.loss_type == 'p_listmle':
                cand_labels = torch.zeros_like(cand_logits)
                cand_labels[:, 0] = 1.0
                loss = p_listmle_loss(cand_logits, cand_labels)
            elif args.loss_type == 'mmcl':
                cand_labels = torch.zeros_like(cand_logits)
                cand_labels[:, 0] = 1.0
                loss = mmcl_loss(cand_logits, cand_labels)
            elif args.loss_type == 'gbce':
                cand_labels = torch.zeros_like(cand_logits)
                cand_labels[:, 0] = 1.0
                loss = gbce_loss(cand_logits, cand_labels)
            elif args.loss_type == 'p_gbce':
                cand_labels = torch.zeros_like(cand_logits)
                cand_labels[:, 0] = 1.0
                loss = p_gbce_loss(cand_logits, cand_labels)
            else:
                loss = (-F.log_softmax(cand_logits, dim=1)[:, 0]).mean()

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / max(1, num_batches)
        print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f}', end='')
        
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
                torch.save(model.state_dict(), os.path.join(args.dataset + '_' + args.train_dir, 'SASRec.best.pth'))
                print('         Γ£ô Saved best model')
            
            f.write(f'{epoch} {t_valid} {t_test}\n')
            f.flush()
            t0 = time.time()
        else:
            print()
    
    f.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best Val  - NDCG@10: {best_val_ndcg:.4f}, HR@10: {best_val_hr:.4f}")
    print(f"Best Test - NDCG@10: {best_test_ndcg:.4f}, HR@10: {best_test_hr:.4f}")
    print("=" * 60)
