import os
import torch
import argparse
from tqdm import tqdm
from collections import Counter

from models.model import SASRec
from models.titans_memory import TitansSASRec
from utils import data_partition

# ── Type coercions matching argparse definitions in main.py ──────────────────
_BOOL_TRUE = {"true", "1", "yes"}
_ARG_TYPES: dict[str, type] = {
    "batch_size": int, "lr": float, "maxlen": int, "hidden_units": int,
    "num_blocks": int, "num_epochs": int, "num_heads": int,
    "dropout_rate": float, "num_negatives": int, "grad_clip": float,
    "warmup_ratio": float, "titans_d_mem": int, "titans_mem_lr_scale": float,
    "titans_mem_wd": float, "titans_mem_epochs": int, "cl_lambda": float,
    "cl_temperature": float, "cosine_restarts": int, "full_ce_weight": float,
    "label_smoothing": float, "gbce_alpha": float,
    "seed": int, "mem_maxlen": int, "mem_start_epoch": int,
    "titans_base_lr_scale": float,
}
_BOOL_ARGS = {
    "inference_only", "norm_first", "use_nested_learning", "use_duorec",
    "cosine_anneal",
}

# Keys removed from the current codebase; silently ignored when parsing
# legacy args.txt files so they don't overwrite the Namespace with junk.
_DEPRECATED_ARGS = {"use_graph", "graph_cache", "graph_reg"}


def load_args_from_txt(args_txt_path: str) -> argparse.Namespace:
    """Parse the args.txt written by main.py and return a Namespace."""
    if not os.path.exists(args_txt_path):
        raise FileNotFoundError(f"args.txt not found: {args_txt_path}")

    ns: dict = {}
    with open(args_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, raw = line.partition(",")
            if key in _DEPRECATED_ARGS:
                continue
            if key in _BOOL_ARGS:
                ns[key] = raw.lower() in _BOOL_TRUE
            elif key in _ARG_TYPES:
                ns[key] = _ARG_TYPES[key](raw)
            else:
                ns[key] = raw  # keep as string (dataset, train_dir, device, …)

    return argparse.Namespace(**ns)


def build_model(args: argparse.Namespace, usernum: int, itemnum: int) -> torch.nn.Module:
    model = SASRec(usernum, itemnum, args).to(args.device)

    if args.use_nested_learning:
        print("   -> Wrapping TitansSASRec…")
        model = TitansSASRec(
            base_model=model,
            d_model=args.hidden_units,
            d_mem=args.titans_d_mem,
            dropout=args.dropout_rate,
            maxlen=args.maxlen,
        ).to(args.device)

    return model


def check_hallucination(checkpoint_path: str, device: str | None = None, top_k: int = 10):
    # ── 1. Locate args.txt next to the checkpoint ───────────────────────────
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    args_txt = os.path.join(ckpt_dir, "args.txt")

    print(f"Loading config from: {args_txt}")
    args = load_args_from_txt(args_txt)

    # Allow caller to override device (e.g. "cpu" for machines without GPU)
    if device is not None:
        args.device = device

    print(f"Dataset : {args.dataset}  |  maxlen={args.maxlen}  |  "
          f"mem_maxlen={args.mem_maxlen}  |  hidden={args.hidden_units}")
    print(f"Flags   : use_nested_learning={args.use_nested_learning}  "
          f"norm_first={args.norm_first}")

    # ── 2. Load dataset ─────────────────────────────────────────────────────
    print("\nLoading dataset partition…")
    user_train, user_valid, user_test, usernum, itemnum = data_partition(args.dataset)

    # ── 3. Build & load model ────────────────────────────────────────────────
    print("Building model…")
    model = build_model(args, usernum, itemnum)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    model.eval()

    # ── 4. Inference loop ────────────────────────────────────────────────────
    use_mem = args.use_nested_learning and args.mem_maxlen > 0

    repetition_count = 0
    total_recs = 0
    all_recommended: list[int] = []

    check_users = list(user_train.keys())
    print(f"\nRunning hallucination check on {len(check_users)} users (top-{top_k})…\n")

    for uid in tqdm(check_users):
        seq = user_train[uid]
        if not seq:
            continue

        # ── Attention sequence (maxlen) ───────────────────────────────────
        seq_t = torch.tensor([seq], dtype=torch.long, device=args.device)
        if seq_t.shape[1] > args.maxlen:
            seq_t = seq_t[:, -args.maxlen:]
        else:
            pad = torch.zeros(1, args.maxlen - seq_t.shape[1], dtype=torch.long, device=args.device)
            seq_t = torch.cat([pad, seq_t], dim=1)

        # ── Memory sequence (mem_maxlen) ──────────────────────────────────
        mem_seq = None
        if use_mem:
            full_t = torch.tensor([seq], dtype=torch.long, device=args.device)
            if full_t.shape[1] > args.mem_maxlen:
                mem_seq = full_t[:, -args.mem_maxlen:]
            else:
                pad_m = torch.zeros(1, args.mem_maxlen - full_t.shape[1], dtype=torch.long, device=args.device)
                mem_seq = torch.cat([pad_m, full_t], dim=1)

        with torch.no_grad():
            item_idx = torch.arange(1, itemnum + 1, device=args.device).unsqueeze(0)
            predict_kwargs = {"mem_seqs": mem_seq} if mem_seq is not None else {}
            logits = model.predict(torch.tensor([uid], device=args.device), seq_t, item_idx, **predict_kwargs)

            if torch.isnan(logits).any():
                print(f"[WARN] NaN logits for user {uid} — skipping")
                continue

            # Mask training history (mirror test-time behaviour)
            for item in seq:
                if 1 <= item <= itemnum:
                    logits[0, item - 1] = -float("inf")

            _, indices = torch.topk(logits, top_k)
            recs = (indices[0].cpu().numpy() + 1).tolist()

        all_recommended.extend(recs)
        repetition_count += len(set(recs).intersection(set(seq)))
        total_recs += top_k

    # ── 5. Report ─────────────────────────────────────────────────────────────
    unique_items = len(set(all_recommended))
    coverage = unique_items / itemnum * 100
    top5 = Counter(all_recommended).most_common(5)
    n_users = len(check_users)

    print("\n" + "=" * 50)
    print("HALLUCINATION CHECK REPORT")
    print("=" * 50)

    print(f"\n[1] Coverage & Diversity")
    print(f"    Unique items recommended : {unique_items}/{itemnum} ({coverage:.2f}%)")
    if coverage < 1.0:
        print("    ❌ Model collapse — only a handful of items are ever recommended.")
    else:
        print("    ✅ Healthy diversity in recommendations.")

    print(f"\n[2] Popularity Bias  (top-5 most recommended items)")
    for item_id, cnt in top5:
        print(f"    Item {item_id:6d} : {cnt:5d}/{n_users} users  ({cnt/n_users*100:.1f}%)")
    if top5[0][1] > n_users * 0.9:
        print("    ⚠️  Item recommended to >90 % of users — severe popularity bias.")
    else:
        print("    ✅ Distribution looks reasonable.")

    print(f"\n[3] Repetition / Hallucination  (history leakage after masking)")
    print(f"    History items in top-{top_k} : {repetition_count}/{total_recs}")
    if repetition_count == 0:
        print("    ✅ No history leakage detected.")
    else:
        print("    ⚠️  History items leaked — verify masking logic.")

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to .pth checkpoint file")
    p.add_argument("--device", default=None, help="Override device (e.g. cpu, cuda:1)")
    p.add_argument("--top_k", default=10, type=int, help="Recommendation list size")
    cli = p.parse_args()

    check_hallucination(cli.checkpoint, device=cli.device, top_k=cli.top_k)