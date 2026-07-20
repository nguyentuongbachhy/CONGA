"""
Long-term Memory & Diversity Analysis for CONGA vs Baselines.

Checkpoint spec format: "name:path:titans:no_rope:no_mhc:no_swiglu"
  titans  : 1 = wrap with TitansSASRec, 0 = plain SASRec
  no_rope : 1 = disable RoPE,   0 = enable (default 0)
  no_mhc  : 1 = disable KromHC, 0 = enable (default 0)
  no_swiglu: 1 = disable SwiGLU, 0 = enable (default 0)

Examples:
  "conga_base:ml-1m_abl_A0/SASRec.best.pth:0:1:1:1"   # A0: no components
  "conga_a3:ml-1m_abl_A3/SASRec.best.pth:0:0:0:0"     # A3: all except titans
  "conga_full:ml-1m_abl_A4/SASRec.best.pth:1:0:0:0"   # A4: full CONGA

Usage:
  python analyze_forgetting.py \
    --dataset ml-1m \
    --checkpoints \
      "conga_base:ablation/ml-1m_abl_A0/SASRec.best.pth:0:1:1:1" \
      "conga_a3:ablation/ml-1m_abl_A3/SASRec.best.pth:0:0:0:0" \
      "conga_full:ablation/ml-1m_abl_A4/SASRec.best.pth:1:0:0:0" \
    --bins 50 150 --topk 10 --device cuda
"""

import os
import argparse
import types
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from utils import data_partition, load_metadata, check_and_convert_dataset
from models.model import SASRec


# ════════════════════════════════════════════════════════════════════════════
# Args
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="name:path:titans:no_rope:no_mhc:no_swiglu")
    p.add_argument("--topk",         default=10,   type=int)
    p.add_argument("--bins",         nargs="+",    type=int, default=[50, 150])
    p.add_argument("--maxlen",       default=300,  type=int)
    p.add_argument("--hidden_units", default=64,   type=int)
    p.add_argument("--num_blocks",   default=2,    type=int)
    p.add_argument("--num_heads",    default=1,    type=int)
    p.add_argument("--dropout_rate", default=0.2,  type=float)
    p.add_argument("--num_streams",  default=4,    type=int)
    p.add_argument("--titans_d_mem", default=128,  type=int)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--batch_size",   default=256,  type=int)
    p.add_argument("--output",       default="forgetting_analysis.txt")
    return p.parse_args()


def parse_checkpoint_spec(spec):
    """Parse 'name:path:titans:no_rope:no_mhc:no_swiglu' with safe defaults."""
    parts = spec.split(":")
    name     = parts[0]
    path     = parts[1]
    titans   = int(parts[2]) if len(parts) > 2 else 0
    no_rope  = int(parts[3]) if len(parts) > 3 else 0
    no_mhc   = int(parts[4]) if len(parts) > 4 else 0
    no_swiglu= int(parts[5]) if len(parts) > 5 else 0
    return name, path, bool(titans), bool(no_rope), bool(no_mhc), bool(no_swiglu)


# ════════════════════════════════════════════════════════════════════════════
# Model loader — per-model architecture flags
# ════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, usernum, itemnum, args,
               use_titans=False, no_rope=False, no_mhc=False, no_swiglu=False):
    fake = types.SimpleNamespace(
        device=args.device,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        maxlen=args.maxlen,
        num_streams=args.num_streams,
        norm_first=False,
        no_rope=no_rope,
        no_mhc=no_mhc,
        no_swiglu=no_swiglu,
        use_freq=False,
        fft_cutoff=3,
        freq_mode="post",
        freq_alpha=0.1,
    )
    model = SASRec(usernum, itemnum, fake).to(args.device)

    if use_titans:
        from models.titans_memory import TitansSASRec
        model = TitansSASRec(
            base_model=model,
            d_model=args.hidden_units,
            d_mem=args.titans_d_mem,
            dropout=args.dropout_rate,
            maxlen=args.maxlen,
        ).to(args.device)

    state = torch.load(ckpt_path, map_location=args.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")

    model.eval()
    return model


def get_item_emb(model, use_titans):
    with torch.no_grad():
        src = model.base_model if use_titans else model
        return src.item_emb.weight.cpu().float().numpy()


# ════════════════════════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def get_predictions(model, user_train, user_valid, user_test, itemnum, args):
    users = [u for u in user_test if len(user_test[u]) > 0]
    preds = {}

    for start in tqdm(range(0, len(users), args.batch_size), desc="  Inference", ncols=80):
        batch = users[start: start + args.batch_size]
        seqs = []
        mask_rows = []
        mask_cols = []

        for i, u in enumerate(batch):
            history = user_train[u] + user_valid[u]
            seq = history[-args.maxlen:]
            pad = [0] * (args.maxlen - len(seq))
            seqs.append(pad + seq)
            
            # Lọc các item hợp lệ (<= itemnum) ngay từ đầu để tránh out-of-bounds
            exclude = set(history)
            valid_exclude = [item for item in exclude if item <= itemnum]
            
            mask_rows.extend([i] * len(valid_exclude))
            mask_cols.extend(valid_exclude)

        seq_t = torch.tensor(seqs, dtype=torch.long, device=args.device)
        feats = model.log2feats(seq_t)[:, -1, :]   # (B, H)

        # Get item embeddings from correct submodule
        src = model.base_model if hasattr(model, "base_model") else model
        logits = feats @ src.item_emb.weight.T      # (B, V+1)

        # Masking padding item (0)
        logits[:, 0] = -1e9
        
        # Vectorized masking cho lịch sử giao dịch (loại bỏ lặp CPU-GPU sync)
        if mask_rows:
            logits[torch.tensor(mask_rows, device=args.device), 
                   torch.tensor(mask_cols, device=args.device)] = -1e9

        topk_ids = torch.topk(logits, args.topk, dim=-1).indices.cpu().numpy()
        for i, u in enumerate(batch):
            preds[u] = topk_ids[i]

    return preds


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

def ndcg_hr_at_k(ranked, gt_set, k):
    for rank, item in enumerate(ranked[:k]):
        if item in gt_set:
            return 1.0 / np.log2(rank + 2), 1
    return 0.0, 0

def ranking_metrics(preds, user_test, k):
    ndcg, hr = [], []
    for u, ranked in preds.items():
        if not user_test.get(u):
            continue
        n, h = ndcg_hr_at_k(ranked, set(user_test[u]), k)
        ndcg.append(n); hr.append(h)
    return np.mean(ndcg), np.mean(hr), len(ndcg)

def duplicate_rate(preds):
    if not preds: return 0.0
    ranked_matrix = np.stack(list(preds.values()))
    sorted_matrix = np.sort(ranked_matrix, axis=1)
    has_dup = (sorted_matrix[:, 1:] == sorted_matrix[:, :-1]).any(axis=1)
    return float(has_dup.mean())

def catalog_coverage(preds, itemnum):
    if not preds: return 0.0
    ranked_matrix = np.stack(list(preds.values()))
    unique_items = np.unique(ranked_matrix)
    return len(unique_items) / itemnum

def intra_list_diversity(preds, emb_np):
    if not preds: return 0.0
    ranked_matrix = np.stack(list(preds.values()))
    k = ranked_matrix.shape[1]
    if k < 2: return 0.0

    norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-9
    normed = emb_np / norms
    
    vecs = normed[ranked_matrix]
    sim = np.einsum('ukh,uvh->ukv', vecs, vecs)
    
    mask = ~np.eye(k, dtype=bool)
    avg_sim_per_user = sim[:, mask].mean(axis=1)
    
    return float(1.0 - avg_sim_per_user.mean())

def tail_item_ratio(preds, item_freq, itemnum, threshold_pct=20):
    if not preds: return 0.0
    counts = np.array(list(item_freq.values()))
    if len(counts) == 0: return 0.0
    cutoff = np.percentile(counts, threshold_pct)
    
    is_tail = np.zeros(itemnum + 1, dtype=bool)
    for item, count in item_freq.items():
        if count <= cutoff:
            is_tail[item] = True
            
    ranked_matrix = np.stack(list(preds.values()))
    tail_hits = is_tail[ranked_matrix].sum()
    
    return float(tail_hits / max(ranked_matrix.size, 1))


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)

    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, _, _ = dataset

    # Item popularity (for tail ratio)
    item_freq = defaultdict(int)
    for u in user_train:
        for item in user_train[u]:
            item_freq[item] += 1

    # Sequence lengths
    seq_lengths = {
        u: len(user_train[u]) + len(user_valid[u])
        for u in user_test if len(user_test[u]) > 0
    }

    # Build groups
    boundaries = sorted(args.bins)
    group_labels = [f"Short  (≤{boundaries[0]})"]
    for i in range(len(boundaries) - 1):
        group_labels.append(f"Medium ({boundaries[i]+1}–{boundaries[i+1]})")
    group_labels.append(f"Long   (>{boundaries[-1]})")

    def get_group(l):
        for i, b in enumerate(boundaries):
            if l <= b: return i
        return len(boundaries)

    user_groups = defaultdict(list)
    for u, l in seq_lengths.items():
        user_groups[get_group(l)].append(u)

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    log("=" * 72)
    log(f"Dataset: {args.dataset}  |  Users: {usernum}  |  Items: {itemnum}")
    log(f"Top-K: {args.topk}  |  Bins: {boundaries}")
    log()
    log("User distribution:")
    for g, label in enumerate(group_labels):
        n = len(user_groups[g])
        log(f"  {label}: {n} users ({100*n/len(seq_lengths):.1f}%)")
    log()

    all_results = {}

    for spec in args.checkpoints:
        name, path, use_titans, no_rope, no_mhc, no_swiglu = parse_checkpoint_spec(spec)

        log("─" * 72)
        log(f"Model: {name}")
        log(f"  Path:    {path}")
        log(f"  Config:  titans={use_titans}  rope={not no_rope}  "
            f"mhc={not no_mhc}  swiglu={not no_swiglu}")

        model = load_model(path, usernum, itemnum, args,
                           use_titans, no_rope, no_mhc, no_swiglu)
        preds = get_predictions(model, user_train, user_valid, user_test, itemnum, args)
        emb_np = get_item_emb(model, use_titans)

        ndcg_all, hr_all, n_all = ranking_metrics(preds, user_test, args.topk)

        # Sanity check — warn if near-random
        if ndcg_all < 0.01:
            log(f"  [WARN] NDCG@{args.topk}={ndcg_all:.4f} is near-random. "
                f"Check architecture flags match training config!")

        dup  = duplicate_rate(preds)
        cov  = catalog_coverage(preds, itemnum)
        ild  = intra_list_diversity(preds, emb_np)
        tail = tail_item_ratio(preds, item_freq, itemnum)

        log()
        log(f"  Overall — NDCG@{args.topk}: {ndcg_all:.4f}  HR@{args.topk}: {hr_all:.4f}  (n={n_all})")
        log(f"  Duplicate Rate    : {dup:.6f}  ({'✓ clean' if dup < 1e-6 else '✗ duplicates present'})")
        log(f"  Catalog Coverage  : {cov:.4f}  ({100*cov:.1f}% of {itemnum} items)")
        log(f"  Intra-List Diversity (ILD): {ild:.4f}")
        log(f"  Tail Item Ratio   : {tail:.4f}  (bottom-20% popularity items)")
        log()

        log(f"  {'Group':<25} {'Users':>6}  {'NDCG@'+str(args.topk):>10}  {'HR@'+str(args.topk):>10}")
        log(f"  {'─'*57}")

        group_results = {}
        for g, label in enumerate(group_labels):
            g_preds = {u: preds[u] for u in user_groups[g] if u in preds}
            n, h, cnt = ranking_metrics(g_preds, user_test, args.topk)
            group_results[label] = (n, h, cnt)
            log(f"  {label:<25} {cnt:>6}  {n:>10.4f}  {h:>10.4f}")

        all_results[name] = {
            "overall": (ndcg_all, hr_all),
            "groups": group_results,
            "dup": dup, "cov": cov, "ild": ild, "tail": tail,
        }
        log()

    # ── Cross-model comparison ────────────────────────────────────────────
    if len(all_results) > 1:
        names = list(all_results.keys())
        log("=" * 72)
        log(f"CROSS-MODEL — NDCG@{args.topk} by sequence length group")
        log("=" * 72)

        col = 12
        header = f"  {'Group':<25}" + "".join(f"  {n:>{col}}" for n in names)
        log(header)
        log("  " + "─" * (25 + (col + 2) * len(names)))

        for label in group_labels:
            row = f"  {label:<25}"
            for name in names:
                n, _, _ = all_results[name]["groups"].get(label, (0, 0, 0))
                row += f"  {n:>{col}.4f}"
            log(row)

        log()

        # Relative gain of last model vs first
        if len(names) >= 2:
            base, comp = names[0], names[-1]
            log(f"  Relative gain: {comp} vs {base}")
            log(f"  {'Group':<25}  {'Δ NDCG':>10}  {'Δ%':>8}")
            log(f"  {'─'*47}")
            for label in group_labels:
                n_b = all_results[base]["groups"].get(label, (0,))[0]
                n_c = all_results[comp]["groups"].get(label, (0,))[0]
                delta = n_c - n_b
                pct = (delta / n_b * 100) if n_b > 1e-6 else float("nan")
                log(f"  {label:<25}  {delta:>+10.4f}  {pct:>7.1f}%")

        log()
        log(f"  {'Metric':<25}" + "".join(f"  {n:>{col}}" for n in names))
        log("  " + "─" * (25 + (col + 2) * len(names)))
        for key, label in [("dup","Dup Rate"), ("cov","Coverage"),
                            ("ild","ILD"), ("tail","Tail Ratio")]:
            row = f"  {label:<25}"
            for name in names:
                row += f"  {all_results[name][key]:>{col}.4f}"
            log(row)

    log()
    log("=" * 72)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()