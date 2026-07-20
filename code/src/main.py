import os
import math
import time
import torch
import argparse
import torch.nn.functional as F
import multiprocessing
from tqdm import tqdm

multiprocessing.set_start_method('spawn', force=True)
from models.model import SASRec
from models.titans_memory import TitansSASRec
from utils import check_and_convert_dataset, load_metadata, get_dataloader, data_partition, evaluate, evaluate_valid
from losses import listmle_loss, p_listmle_loss, p_sampled_softmax_loss, mmcl_loss, gbce_loss, p_gbce_loss, rc_gbce_loss, approx_ndcg_loss, infonce_gbce_loss, tcr_loss, composite_loss, duorec_cl_loss


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="conga", type=str,
                    choices=["conga", "bsarec", "bsarec_rope", "sasrec", "bert4rec",
                             "gru4rec", "fmlprec", "duorec", "fearec", "wearec"],
                    help="'conga' uses the CONGA pipeline (titans/losses/etc.). "
                         "Any other value runs the BSARec-style benchmark pipeline.")
parser.add_argument("--dataset", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=200, type=int)
parser.add_argument("--hidden_units", default=64, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--num_negatives", default=20, type=int)
parser.add_argument("--neg_sampling_mode", default="random", type=str, choices=["random", "popularity", "frequency", "mans"])
parser.add_argument("--loss_type", default="gbce", type=str, choices=["ce", "sampled_softmax", "p_sampled_softmax", "listmle", "p_listmle", "mmcl", "gbce", "p_gbce", "rc_gbce", "approx_ndcg", "infonce_gbce", "tcr", "composite"])
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--inference_only", default=False, action="store_true")
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument("--norm_first", action="store_true", default=False)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--grad_clip", default=0.0, type=float, help="Gradient clipping max norm (0 = no clipping)")
parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Fraction of total steps for LR warmup")
parser.add_argument("--use_nested_learning", default=False, action="store_true")
parser.add_argument("--mask_ratio", default=0.0, type=float,
                    help="CONGA: random-masking ratio on input. BERT4Rec: BERT-style mask ratio.")
parser.add_argument("--titans_d_mem", default=128, type=int)
parser.add_argument("--titans_mem_lr_scale", default=1.0, type=float, help="memory LR = lr * scale")
parser.add_argument("--titans_mem_wd", default=0.01, type=float, help="weight decay for memory params")
parser.add_argument("--titans_mem_epochs", default=0, type=int, help="memory cosine epochs (0=same as total)")
parser.add_argument("--use_duorec", default=False, action="store_true")
parser.add_argument("--no_amp", action="store_true", default=False,
                    help="Disable bfloat16 autocast (train in float32 like benchmark_runner).")
parser.add_argument("--cl_lambda", default=0.1, type=float)
parser.add_argument("--cl_temperature", default=0.1, type=float)
parser.add_argument("--cosine_anneal", default=False, action="store_true")
parser.add_argument("--cosine_restarts", default=0, type=int, help="Number of cosine warm restarts (0=single cycle)")
parser.add_argument("--full_ce_weight", default=0.0, type=float)
parser.add_argument("--label_smoothing", default=0.0, type=float)
parser.add_argument("--gbce_alpha", default=0.0, type=float, help="GBCE alpha (0=auto)")
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
parser.add_argument("--mem_maxlen", default=0, type=int, help="Extended sequence length for memory (0=same as maxlen)")
parser.add_argument("--mem_start_epoch", default=0, type=int, help="Epoch to start memory fusion (0=from beginning). Base model is frozen after this epoch.")
parser.add_argument("--titans_base_lr_scale", default=0.0, type=float, help="Phase 2 base model LR = lr * scale (0=frozen)")
parser.add_argument("--no_rope", action="store_true", default=False)
parser.add_argument("--no_mhc", action="store_true", default=False)
parser.add_argument("--no_swiglu", action="store_true", default=False)
parser.add_argument("--num_streams", default=4, type=int,
                    help="Number of MHCv2 streams (must be power of 2: 2, 4, 8).")
parser.add_argument("--use_freq", action="store_true", default=False,
                    help="Enable BSARec-style FFT frequency filter blended inside KromHC layer_fn.")
parser.add_argument("--fft_cutoff", default=3, type=int,
                    help="FFT low-pass cutoff (number of kept frequency bins, like BSARec's --c//2+1).")
parser.add_argument("--freq_alpha", default=0.1, type=float,
                    help="Fixed blend weight for FFT path vs attention path (BSARec alpha). Default=0.1.")
parser.add_argument("--freq_mode", default="post", choices=["post", "parallel"],
                    help="'post': single FrequencyLayer after all blocks (v4). "
                         "'parallel': per-block FrequencyLayer parallel to attention (BSARec style).")

# ---------------------------------------------------------------------
# BSARec-style benchmark hyperparameters (only used when --model_type
# is NOT 'conga'). Defaults match BSARec/src/utils.py; per-dataset best
# values live in benchmark_runner.BEST_CONFIGS and are applied unless
# the user overrides them on the command line.
# ---------------------------------------------------------------------
parser.add_argument("--patience", default=10, type=int,
                    help="Benchmark early-stopping patience (eval cycles).")
parser.add_argument("--eval_every", default=10, type=int,
                    help="Evaluate every N epochs (Phase 1 / no-TITANS).")
parser.add_argument("--phase2_eval_every", default=0, type=int,
                    help="Evaluate every N epochs during Phase 2 (TITANS active). 0=use same as --eval_every.")
parser.add_argument("--phase2_num_epochs", default=0, type=int,
                    help="Max epochs to run in Phase 2 / TITANS phase after activation (0=run until --num_epochs).")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Benchmark Adam weight decay.")
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--attention_probs_dropout_prob", default=-1.0, type=float,
                    help="Benchmark attention dropout (-1 = fall back to --dropout_rate).")
parser.add_argument("--hidden_act", default="gelu", type=str,
                    choices=["gelu", "relu", "swish", "tanh", "sigmoid"])
parser.add_argument("--initializer_range", default=0.02, type=float)
# BSARec
parser.add_argument("--alpha", default=0.7, type=float, help="BSARec frequency-vs-attention mix / WEARec wavelet-vs-FFT blend.")
parser.add_argument("--c", default=5, type=int, help="BSARec low-pass cutoff.")
# DuoRec / FEARec
parser.add_argument("--tau", default=1.0, type=float)
parser.add_argument("--lmd", default=0.1, type=float)
parser.add_argument("--lmd_sem", default=0.1, type=float)
parser.add_argument("--ssl", default="us_x", type=str, choices=["us", "un", "su", "us_x", "none"])
parser.add_argument("--sim", default="dot", type=str, choices=["dot", "cos"])
# FEARec
parser.add_argument("--spatial_ratio", default=0.1, type=float)
parser.add_argument("--global_ratio", default=0.6, type=float)
parser.add_argument("--fredom", default="True", type=str, help="FEARec: 'True' or 'False'.")
parser.add_argument("--fredom_type", default="us_x", type=str)
# GRU4Rec
parser.add_argument("--gru_hidden_size", default=64, type=int)

args = parser.parse_args()

# Track which BSARec-style args the user actually overrode so we don't
# clobber them with the dataset-specific best preset.
_BENCH_OVERRIDABLE = {
    "lr", "batch_size", "num_epochs", "maxlen", "hidden_units", "num_blocks",
    "num_heads", "dropout_rate", "attention_probs_dropout_prob", "hidden_act",
    "initializer_range", "alpha", "c", "mask_ratio", "loss_type",
    "tau", "lmd", "lmd_sem", "ssl", "sim", "spatial_ratio", "global_ratio",
    "fredom", "fredom_type", "gru_hidden_size", "weight_decay",
    "adam_beta1", "adam_beta2",
    "num_streams", "no_rope",
}
_cli_overrides = {
    k for k in _BENCH_OVERRIDABLE
    if getattr(args, k, None) != parser.get_default(k)
}

if __name__ == "__main__":
    # Branch to the BSARec-style benchmark pipeline. This keeps the
    # CONGA training code below completely unchanged for 'conga'.
    if args.model_type.lower() != "conga":
        from benchmark_runner import run_benchmark
        run_benchmark(args, cli_overrides=_cli_overrides)
        raise SystemExit(0)

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.dataset + "_" + args.train_dir, exist_ok=True)
    
    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)
    
    # --- Adaptive num_streams / RoPE based on dataset density ---
    # Load sequence-length stats once; reused by both heuristics.
    _user_idx = np.load(os.path.join("bins", f"{args.dataset}_bin", "user_index.npy"))
    _lengths = _user_idx[_user_idx[:, 1] > 0, 1]
    _avg_len = float(_lengths.mean())

    # num_streams: 4 streams for denser datasets (avg_L >= 50), 2 for sparser.
    if 'num_streams' not in _cli_overrides:
        args.num_streams = 4 if _avg_len >= 50 else 2
        print(f"Adaptive num_streams: avg_seq_len={_avg_len:.1f} -> num_streams={args.num_streams}")
    else:
        print(f"num_streams={args.num_streams} (user override)")

    # RoPE: beneficial only on long sequences where absolute PE faces OOD
    # positions. On short sequences (avg_L < 50), absolute PE overfits the
    # narrow position range better — RoPE rotation angles are too similar
    # to carry useful signal. Auto-disable unless user explicitly set --no_rope.
    if 'no_rope' not in _cli_overrides:
        if _avg_len < 50:
            args.no_rope = True
            print(f"Adaptive RoPE: avg_seq_len={_avg_len:.1f} < 50 -> RoPE disabled (use --no_rope=False to override)")
        else:
            print(f"Adaptive RoPE: avg_seq_len={_avg_len:.1f} >= 50 -> RoPE enabled")

    with open(os.path.join(args.dataset + "_" + args.train_dir, "args.txt"), "w") as f_args:
        f_args.write("\n".join([f"{k},{v}" for k, v in sorted(vars(args).items())]))

    tensor_mem_gb = (args.batch_size * args.maxlen * args.num_negatives * args.hidden_units * 4) / (1024**3)
    print(f"Dataset: {args.dataset} | Users: {usernum} | Items: {itemnum}")
    print(f"Estimated VRAM: {tensor_mem_gb:.2f} GB")
    
    if tensor_mem_gb > 8.0:
        print("[WARNING] High VRAM required! Reduce --num_negatives or --batch_size")
    
    print(f"Loss: {args.loss_type} | Neg sampling: {args.neg_sampling_mode}")
    
    # Optimization: TF32 for Ampere+ GPUs
    torch.set_float32_matmul_precision('high')

    initial_mem_maxlen = args.mem_maxlen if (args.use_nested_learning and args.mem_start_epoch == 0) else 0
    args._mem_active = (args.use_nested_learning and initial_mem_maxlen > 0 and args.mem_start_epoch == 0)
    
    train_loader = get_dataloader(
        args.dataset, args.maxlen, args.batch_size, mode="train", 
        num_workers=args.num_workers, num_negatives=args.num_negatives, 
        neg_sampling_mode=args.neg_sampling_mode, mem_maxlen=initial_mem_maxlen,
        use_duorec=args.use_duorec,
    )

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, _, _] = dataset
    
    f = open(os.path.join(args.dataset + "_" + args.train_dir, "log.txt"), "w")
    f.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")
    
    model = SASRec(usernum, itemnum, args).to(args.device)

    # BSARec/DuoRec-style module-based init: Normal(0, 0.02) for Linear +
    # Embedding, ones/zeros for LayerNorm. Previously a blanket
    # `xavier_normal_` over every parameter produced Linear weights with
    # std ~0.08-0.13 (4-6x larger than the BSARec convention), hurting
    # stability on sparse datasets like Beauty / yelp.
    def _init_weights(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    model.apply(_init_weights)

    # Restore specialised inits that the generic loop over-writes.
    torch.nn.init.constant_(model.stream_fusion.weight, 1.0 / model.num_streams)
    model.item_emb.weight.data[0, :] = 0

    if args.use_nested_learning and args.mem_start_epoch == 0:
        print("Applying TITANS neural memory (from epoch 1)...")
        model = TitansSASRec(
            base_model=model,
            d_model=args.hidden_units,
            d_mem=getattr(args, 'titans_d_mem', None),
            dropout=args.dropout_rate,
            maxlen=args.maxlen,
        ).to(args.device)
        print(f"✓ TITANS applied | d_model={args.hidden_units}, d_mem={model.memory.d_mem}")
    
    epoch_start_idx = 1
    if args.state_dict_path:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            epoch_marker = args.state_dict_path.find("epoch=")
            if epoch_marker != -1:
                tail = args.state_dict_path[epoch_marker + 6:]
                epoch_start_idx = int(tail[:tail.find(".")]) + 1
                print(f"Loaded checkpoint from epoch {epoch_start_idx - 1}")
            else:
                print(f"Loaded checkpoint: {args.state_dict_path}")
        except Exception as e:
            print(f"Failed loading checkpoint: {e}")
    
    if args.inference_only:
        print("[Inference Only]")
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f"Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})")
        exit(0)
    
    if args.use_nested_learning:
        if args.mem_start_epoch > 0:
            # Two-phase training: Phase 1 = pure SASRec (no Titans wrapper)
            # This ensures Phase 1 is IDENTICAL to the pure baseline 
            # (same parameter initialization, same random state, same optimizer)
            print(f"TITANS: deferred to Phase 2 (epoch {args.mem_start_epoch})")
            args._titans_config = {
                'd_model': args.hidden_units,
                'd_mem': getattr(args, 'titans_d_mem', None),
                'dropout': args.dropout_rate,
                'maxlen': args.maxlen,
            }
            # Phase 1: pure SASRec optimizer (no memory params)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
        else:
            # Phase 2/3 or joint training (model already wrapped as TitansSASRec)
            mem_lr = args.lr * args.titans_mem_lr_scale
            base_lr_scale = getattr(args, 'titans_base_lr_scale', 0.0)
            mem_params = list(model.memory.parameters()) + list(model.mem_pos_emb.parameters())
            param_groups = [{"params": mem_params, "lr": mem_lr, "weight_decay": args.titans_mem_wd, "name": "titans_memory"}]
            if base_lr_scale > 0:
                model.base_model.requires_grad_(True)
                base_params = list(model.base_model.parameters())
                param_groups.append({"params": base_params, "lr": args.lr * base_lr_scale, "name": "base_model"})
                print(f"Base model unfrozen: lr={args.lr * base_lr_scale:.6f}")
            else:
                model.base_model.requires_grad_(False)
                print("Base model frozen.")
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), weight_decay=0.01)
            print(f"TITANS optimizer: memory_params={sum(p.numel() for p in mem_params)}, mem_lr={mem_lr:.6f}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    total_steps = args.num_epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = None
    if warmup_steps > 0 or args.cosine_anneal:
        from torch.optim.lr_scheduler import LambdaLR
        n_restarts = getattr(args, 'cosine_restarts', 0)
        def make_lr_lambda(total_s, warmup_s):
            def lr_lambda(step):
                if step < warmup_s:
                    return float(step) / float(max(1, warmup_s))
                if args.cosine_anneal:
                    remaining = total_s - warmup_s
                    elapsed = step - warmup_s
                    if n_restarts > 0:
                        cycle_len = remaining // (n_restarts + 1)
                        cycle = min(elapsed // cycle_len, n_restarts)
                        cycle_elapsed = elapsed - cycle * cycle_len
                        cycle_progress = cycle_elapsed / max(1, cycle_len)
                        decay = 0.5 ** cycle
                        return decay * 0.5 * (1.0 + math.cos(math.pi * min(cycle_progress, 1.0)))
                    progress = min(elapsed / max(1, remaining), 1.0)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                return 1.0
            return lr_lambda
        
        base_lambda = make_lr_lambda(total_steps, warmup_steps)
        
        # Per-group scheduling: memory can have shorter cosine cycle
        if args.use_nested_learning and args.titans_mem_epochs > 0:
            mem_total_steps = args.titans_mem_epochs * len(train_loader)
            mem_warmup_steps = int(mem_total_steps * args.warmup_ratio)
            mem_lambda = make_lr_lambda(mem_total_steps, mem_warmup_steps)
            # param_groups order: [memory, base]
            scheduler = LambdaLR(optimizer, [mem_lambda, base_lambda])
            print(f"Memory LR: cosine over {args.titans_mem_epochs} epochs ({mem_total_steps} steps)")
        else:
            scheduler = LambdaLR(optimizer, base_lambda)
        parts = []
        if warmup_steps > 0:
            parts.append(f"warmup={warmup_steps}")
        if args.cosine_anneal:
            parts.append("cosine")
        print(f"LR schedule: {', '.join(parts)} (total={total_steps})")
    
    if args.grad_clip > 0:
        print(f"Gradient clipping: max_norm={args.grad_clip}")
    
    best_val_ndcg5, best_val_hr5, best_val_ndcg10, best_val_hr10 = 0.0, 0.0, 0.0, 0.0
    best_test_ndcg5, best_test_hr5, best_test_ndcg10, best_test_hr10 = 0.0, 0.0, 0.0, 0.0
    T, t0 = 0.0, time.time()
    _titans_active = args.use_nested_learning and args.mem_start_epoch == 0

    auto_phase = args.use_nested_learning and args.mem_start_epoch == 9999
    patience_limit = args.patience
    no_improve_cnt = 0
    current_phase = 1
    _phase2_start_epoch = None  # set when TITANS activates
    if auto_phase:
        print(f"Auto-Phase Transition enabled (patience={patience_limit})")

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        
        # Two-phase training: determine if this epoch uses memory
        use_mem_this_epoch = (
            args.use_nested_learning 
            and args.mem_maxlen > 0
            and (args.mem_start_epoch == 0 or epoch >= args.mem_start_epoch)
        )

        # Phase transition: wrap model with Titans and freeze base
        if (args.use_nested_learning and args.mem_start_epoch > 0 
            and epoch == args.mem_start_epoch):
            print(f"\n{'='*60}")
            print(f"PHASE 2: Creating Titans memory")
            print(f"{'='*60}")
            
            cfg = args._titans_config
            model = TitansSASRec(
                base_model=model,
                d_model=cfg['d_model'],
                d_mem=cfg['d_mem'],
                dropout=cfg['dropout'],
                maxlen=cfg['maxlen'],
            ).to(args.device)
            print(f"✓ TITANS applied | d_model={cfg['d_model']}, d_mem={model.memory.d_mem}")
            
            # Phase 2 optimizer: memory params + optionally fine-tune base
            mem_params = list(model.memory.parameters()) + list(model.mem_pos_emb.parameters())
            mem_lr = args.lr * args.titans_mem_lr_scale
            base_lr_scale = getattr(args, 'titans_base_lr_scale', 0.0)
            
            if base_lr_scale > 0:
                # Unfreeze base with very low LR for co-adaptation
                base_params = list(model.base_model.parameters())
                base_lr = args.lr * base_lr_scale
                optimizer = torch.optim.AdamW([
                    {"params": mem_params, "lr": mem_lr, "weight_decay": args.titans_mem_wd},
                    {"params": base_params, "lr": base_lr, "weight_decay": 0.01},
                ], betas=(0.9, 0.98))
                print(f"Base model UNFROZEN | base_lr={base_lr:.6f} ({base_lr_scale}x)")
                print(f"Base params: {sum(p.numel() for p in base_params)}")
            else:
                # Freeze base model parameters
                for p in model.base_model.parameters():
                    p.requires_grad_(False)
                optimizer = torch.optim.AdamW(
                    [{"params": mem_params, "lr": mem_lr, "weight_decay": args.titans_mem_wd}],
                    betas=(0.9, 0.98),
                )
                print(f"Base model FROZEN")
            
            # Scheduler for phase 2
            remaining = args.num_epochs - args.mem_start_epoch + 1
            total_steps_p2 = remaining * len(train_loader)
            warmup_p2 = int(0.05 * total_steps_p2)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_p2),
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_p2 - warmup_p2),
                ],
                milestones=[warmup_p2],
            )
            print(f"Memory params: {sum(p.numel() for p in mem_params)}")
            print(f"Memory LR: {mem_lr:.6f}, Phase 2 steps: {total_steps_p2}, warmup: {warmup_p2}")
            # Enable memory in evaluation
            args._mem_active = True
            _titans_active = True
            _phase2_start_epoch = epoch
            if args.phase2_num_epochs > 0:
                print(f"Phase 2 will run for {args.phase2_num_epochs} epochs (until epoch {epoch + args.phase2_num_epochs - 1})")

        # Stop phase 2 after phase2_num_epochs if set
        if (_phase2_start_epoch is not None and args.phase2_num_epochs > 0
                and epoch >= _phase2_start_epoch + args.phase2_num_epochs):
            print(f"\nPhase 2 completed ({args.phase2_num_epochs} epochs). Training done.")
            break

        # Dynamic eval cadence: phase 2 evaluates more frequently than phase 1
        _p2_eval = args.phase2_eval_every if args.phase2_eval_every > 0 else args.eval_every
        _eval_every = _p2_eval if _titans_active else args.eval_every

        epoch_loss, num_batches = 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", unit="batch", ncols=100)
        
        for step, batch in enumerate(pbar):
            # Unpack batch: 4 standard tensors + optional extended memory sequence
            # batch_tensors = [x.to(args.device, non_blocking=True) for x in batch]
            # u, seq, pos, neg = batch_tensors[0], batch_tensors[1], batch_tensors[2], batch_tensors[3]
            # mem_seq = batch_tensors[4] if len(batch_tensors) > 4 else None

            batch_tensors = [x.to(args.device, non_blocking=True) for x in batch]
            u = batch_tensors[0]
            seq = batch_tensors[1]
            pos = batch_tensors[2]

            # Tuple order emitted by SASRecDataset:
            #   (uid, seq, pos, [sem_seq if use_duorec], [mem_seq if mem_maxlen>0])
            # sem_seq sits BEFORE mem_seq so the two extras can be independent.
            _cursor = 3
            if args.use_duorec:
                sem_seq = batch_tensors[_cursor]
                _cursor += 1
            else:
                sem_seq = None
            mem_seq = batch_tensors[_cursor] if len(batch_tensors) > _cursor else None

            with torch.no_grad():
                neg = torch.randint(
                    1, itemnum + 1, 
                    (u.size(0), args.maxlen, args.num_negatives), 
                    dtype=torch.long, 
                    device=args.device
                )

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda', enabled=not args.no_amp, dtype=torch.bfloat16):
                mask = pos != 0
                mask_exp = mask.unsqueeze(-1).expand(-1, -1, args.num_negatives)

                input_seq = seq

                if args.mask_ratio > 0 and model.training:
                    noise_mask = torch.rand_like(seq, dtype=torch.float32) < args.mask_ratio
                    noise_mask = noise_mask & (seq != 0)
                    # Safety: never mask ALL tokens for a row — at least 1 non-padding
                    # token must survive. Without this, all-padding input causes NaN in
                    # attention (softmax of all -inf), which propagates to inf gradients
                    # and corrupts Adam momentum (especially harmful on sparse datasets
                    # like Steam where users have very short sequences).
                    seq_len = (seq != 0).sum(dim=1, keepdim=True)  # (B, 1)
                    num_masked = noise_mask.sum(dim=1, keepdim=True)  # (B, 1)
                    would_all_mask = (num_masked >= seq_len)  # (B, 1)
                    # For rows that would be fully masked, keep the last valid token
                    last_valid_pos = (seq != 0).long().cumsum(dim=1).eq(seq_len) & (seq != 0)  # (B, L)
                    noise_mask = noise_mask & ~(would_all_mask & last_valid_pos)
                    input_seq = seq.masked_fill(noise_mask, 0)
                    # Exclude noise-masked positions from loss: those positions have
                    # zero embedding (input_seq=0) which gives rstd=1/sqrt(eps)=1e6 in
                    # MHCv2 LN. A non-zero gradient from CE loss at such a position is
                    # amplified by 1e6 per MHC layer → gradient explosion.
                    mask = mask & ~noise_mask

                pos_logits, neg_logits, log_feats, _ = model(u, input_seq, pos, neg, mem_seqs=mem_seq) if (mem_seq is not None and use_mem_this_epoch) else model(u, input_seq, pos, neg)

                # NaN diagnostic (first 2 batches of epoch 1)
                if epoch == 1 and step < 2:
                    has_nan = torch.isnan(pos_logits).any() or torch.isnan(neg_logits).any()
                # Only build cand_logits for losses that actually need the
                # sampled negatives. CE supervises over the full vocabulary and
                # skips this path (saves O(B*L*K) masked_select + a cat).
                if args.loss_type != "ce":
                    # Note: masked_select creates dynamic shapes which may trigger recompilation if graph capture is strict.
                    # However, usually fine with default torch.compile().
                    pos_sel = torch.masked_select(pos_logits, mask)
                    neg_sel = torch.masked_select(neg_logits, mask).unsqueeze(1) if neg_logits.dim() == 2 else torch.masked_select(neg_logits, mask_exp).view(-1, args.num_negatives)
                    cand_logits = torch.cat([pos_sel.unsqueeze(1), neg_sel], dim=1)

                if args.loss_type == "ce":
                    # DuoRec main loss: full-vocab Cross-Entropy supervising
                    # every non-padding position (SASRec-style). Mirrors
                    # benchmark_models/duorec.py:calculate_loss when
                    # answers.dim() == 2. Combine with --use_duorec to also
                    # enable the unsupervised dropout-view InfoNCE term.
                    #
                    # Speed: gather valid rows BEFORE the matmul so the V*H
                    # GEMM runs on N_valid rows instead of B*L (≈ maxlen /
                    # avg_seq_len speedup, e.g. ~4x on sparse datasets).
                    feats_valid = log_feats[mask]                     # (N_valid, H)
                    targets = pos[mask].long()                         # (N_valid,)
                    logits = F.linear(feats_valid, model.item_emb.weight)  # (N_valid, V)
                    loss = F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)
                elif args.loss_type == "sampled_softmax":
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
                    if args.label_smoothing > 0:
                        ls = args.label_smoothing
                        labels = labels * (1 - ls) + ls / labels.shape[1]
                    alpha = args.gbce_alpha if args.gbce_alpha > 0 else (0.75 if args.num_negatives >= 5 else 0.7)
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

                if args.full_ce_weight > 0 and args.loss_type != "ce":
                    ce_logits = F.linear(log_feats, model.item_emb.weight)
                    valid = (pos != 0)
                    loss = loss + args.full_ce_weight * F.cross_entropy(ce_logits[valid], pos[valid].long())

                if args.use_duorec:
                    ssl_mode = getattr(args, "ssl", "us_x")
                    lmd = args.cl_lambda
                    lmd_sem = getattr(args, "lmd_sem", lmd)
                    temp = args.cl_temperature
                    sem_available = sem_seq is not None

                    if ssl_mode == "us_x":
                        if sem_available:
                            # Batch aug + sem into ONE forward pass (2B instead of B+B)
                            combined_feats = model.log2feats(
                                torch.cat([seq, sem_seq], dim=0)
                            )[:, -1, :]
                            aug_feats, sem_feats = combined_feats.chunk(2, dim=0)
                            loss = loss + lmd_sem * duorec_cl_loss(aug_feats, sem_feats, temp, sim="cos")
                        else:
                            seq_last = log_feats[:, -1, :]
                            aug_feats = model.log2feats(seq)[:, -1, :]
                            loss = loss + lmd * duorec_cl_loss(seq_last, aug_feats, temp, sim="cos")

                    elif ssl_mode in ("us", "un", "su"):
                        seq_last = log_feats[:, -1, :]
                        need_aug = ssl_mode in ("us", "un")
                        need_sem = ssl_mode in ("us", "su") and sem_available

                        if need_aug and need_sem:
                            # Batch both into one forward pass
                            combined_feats = model.log2feats(
                                torch.cat([seq, sem_seq], dim=0)
                            )[:, -1, :]
                            aug_feats, sem_feats = combined_feats.chunk(2, dim=0)
                            loss = loss + lmd * duorec_cl_loss(seq_last, aug_feats, temp, sim="cos")
                            loss = loss + lmd_sem * duorec_cl_loss(seq_last, sem_feats, temp, sim="cos")
                        elif need_aug:
                            aug_feats = model.log2feats(seq)[:, -1, :]
                            loss = loss + lmd * duorec_cl_loss(seq_last, aug_feats, temp, sim="cos")
                        elif need_sem:
                            sem_feats = model.log2feats(sem_seq)[:, -1, :]
                            loss = loss + lmd_sem * duorec_cl_loss(seq_last, sem_feats, temp, sim="cos")

            loss.backward()

            loss_val = loss.item()
            # NaN/Inf guard: skip this step if loss OR gradients are non-finite.
            # Root cause: sparse datasets (e.g. Steam) can produce all-padding batches
            # after mask_ratio masking → NaN attention → inf gradient in MHC backward.
            # Skipping the optimizer step prevents Adam momentum corruption.
            if not torch.isfinite(torch.tensor(loss_val)):
                print(f"\n[WARN] Epoch {epoch} step {step}: non-finite loss={loss_val:.4f}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                grad_norm = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm()

            if not torch.isfinite(grad_norm):
                print(f"\n[WARN] Epoch {epoch} step {step}: inf/NaN gradient norm, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            
            epoch_loss += loss_val
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
        
        avg_loss = epoch_loss / max(1, num_batches)
        gate_info = ""
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}{gate_info}", end="")
        
        if epoch % _eval_every == 0:
            eval_model = model
            eval_model.eval()
            T += time.time() - t0
            with torch.no_grad():
                t_test = evaluate(eval_model, dataset, args)
                t_valid = evaluate_valid(eval_model, dataset, args)

            print(f" | Time: {T:.1f}s")
            print(f"         Valid: N@5={t_valid[0]:.4f}, H@5={t_valid[1]:.4f}, N@10={t_valid[2]:.4f}, H@10={t_valid[3]:.4f}")
            print(f"         Test : N@5={t_test[0]:.4f}, H@5={t_test[1]:.4f}, N@10={t_test[2]:.4f}, H@10={t_test[3]:.4f}")

            if t_valid[2] > best_val_ndcg10:
                best_val_ndcg5, best_val_hr5, best_val_ndcg10, best_val_hr10 = t_valid
                best_test_ndcg5, best_test_hr5, best_test_ndcg10, best_test_hr10 = t_test
                ckpt_name = f"SASRec.phase{current_phase}_best.pth" if auto_phase else "SASRec.best.pth"
                ckpt_path = os.path.join(args.dataset + "_" + args.train_dir, ckpt_name)
                torch.save(model.state_dict(), ckpt_path)
                if auto_phase:
                    torch.save(model.state_dict(), os.path.join(args.dataset + "_" + args.train_dir, "SASRec.best.pth"))
                no_improve_cnt = 0
                print(f"         ✓ Saved best model (phase={current_phase})")
            else:
                if auto_phase:
                    no_improve_cnt += 1

            if auto_phase and no_improve_cnt >= patience_limit:
                if current_phase == 1:
                    print(f"\n{'='*60}\nAuto-Phase: 1 -> 2 (Freeze base, train TITANS)\n{'='*60}")

                    print("Re-initializing DataLoader for Memory (TITANS)...")
                    train_loader = get_dataloader(
                        args.dataset, args.maxlen, args.batch_size, mode="train",
                        num_workers=args.num_workers, num_negatives=args.num_negatives,
                        neg_sampling_mode=args.neg_sampling_mode, mem_maxlen=args.mem_maxlen,
                        use_duorec=args.use_duorec,
                    )

                    p1_path = os.path.join(args.dataset + "_" + args.train_dir, "SASRec.phase1_best.pth")
                    model.load_state_dict(torch.load(p1_path, map_location=args.device))
                    cfg = args._titans_config
                    model = TitansSASRec(base_model=model, d_model=cfg['d_model'], d_mem=cfg['d_mem'],
                                        dropout=cfg['dropout'], maxlen=cfg['maxlen']).to(args.device)
                    model.base_model.requires_grad_(False)
                    mem_params = list(model.memory.parameters()) + list(model.mem_pos_emb.parameters())
                    mem_lr = args.lr * args.titans_mem_lr_scale
                    optimizer = torch.optim.AdamW(
                        [{"params": mem_params, "lr": mem_lr, "weight_decay": args.titans_mem_wd}],
                        betas=(0.9, 0.98),
                    )
                    scheduler = None
                    args.mem_start_epoch = 0
                    args._mem_active = True
                    _titans_active = True
                    current_phase = 2
                    no_improve_cnt = 0
                    best_val_ndcg10 = 0.0
                    _phase2_start_epoch = epoch + 1
                    if args.phase2_num_epochs > 0:
                        print(f"Phase 2 will run for {args.phase2_num_epochs} epochs")
                    print(f"Base frozen | mem_lr={mem_lr:.6f} | mem_params={sum(p.numel() for p in mem_params)}")
                elif current_phase == 2:
                    print(f"\n{'='*60}\nAuto-Phase: 2 -> 3 (Co-finetune)\n{'='*60}")
                    p2_path = os.path.join(args.dataset + "_" + args.train_dir, "SASRec.phase2_best.pth")
                    model.load_state_dict(torch.load(p2_path, map_location=args.device))
                    model.base_model.requires_grad_(True)
                    mem_params = list(model.memory.parameters()) + list(model.mem_pos_emb.parameters())
                    base_params = list(model.base_model.parameters())
                    mem_lr = args.lr * args.titans_mem_lr_scale
                    base_lr = args.lr * 0.01
                    optimizer = torch.optim.AdamW([
                        {"params": mem_params, "lr": mem_lr, "weight_decay": args.titans_mem_wd},
                        {"params": base_params, "lr": base_lr, "weight_decay": 0.01},
                    ], betas=(0.9, 0.98))
                    scheduler = None
                    current_phase = 3
                    no_improve_cnt = 0
                    best_val_ndcg10 = 0.0
                    print(f"Base unfrozen @ lr={base_lr:.6f} | mem_lr={mem_lr:.6f}")
                else:
                    print("\nAuto-Phase: Phase 3 converged. Training complete.")
                    break

            f.write(f"{epoch} {t_valid} {t_test}\n")
            f.flush()
            t0 = time.time()
        else:
            print()
    
    f.close()

    print("\nTraining completed!")
    print(f"Best Val  - N@5: {best_val_ndcg5:.4f}, H@5: {best_val_hr5:.4f}, N@10: {best_val_ndcg10:.4f}, H@10: {best_val_hr10:.4f}")
    print(f"Best Test - N@5: {best_test_ndcg5:.4f}, H@5: {best_test_hr5:.4f}, N@10: {best_test_ndcg10:.4f}, H@10: {best_test_hr10:.4f}")