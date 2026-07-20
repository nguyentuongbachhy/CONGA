#!/usr/bin/env python3
"""Run CONGA + all baseline models across requested datasets.

For baselines (bsarec, sasrec, bert4rec, gru4rec, fmlprec, duorec, fearec,
wearec), the original reference main.py is invoked directly:
  - BSARec paper baselines  → refs/BSARec/src/main.py
  - WEARec model            → refs/WEARec/src/main.py

CONGA's utils are used *only* for data preprocessing (k-core filtering +
re-indexing → BSARec user-sequence .txt format).  CONGA training itself
continues to use CONGA's own main.py.

Configs match the published best hyperparameters from each paper:
  - BSARec (AAAI 2024): lr, alpha, c, num_attention_heads, dropout, maxlen
  - WEARec (AAAI 2026): lr, alpha, num_heads, dropout, maxlen

Usage (from the src/ directory):
    uv run python scripts/run_all_benchmarks.py
    uv run python scripts/run_all_benchmarks.py --datasets Beauty yelp
    uv run python scripts/run_all_benchmarks.py --models conga bsarec sasrec
    uv run python scripts/run_all_benchmarks.py --dry_run
    uv run python scripts/run_all_benchmarks.py --force
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Everything we invoke assumes cwd == src/.
SRC_DIR = Path(__file__).resolve().parent.parent
REFS_DIR = SRC_DIR.parent / "refs"
sys.path.insert(0, str(SRC_DIR))

from benchmark_runner import CONGA_CONFIGS  # noqa: E402
from utils import check_and_convert_dataset, export_bsarec_txt  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASETS = ["Beauty", "ml-1m", "Steam", "yelp"]
BASELINES = ["bsarec", "sasrec", "bert4rec",
             "gru4rec", "fmlprec", "duorec", "fearec", "wearec"]
ALL_MODELS = ["conga"] + BASELINES

# Per-dataset epoch caps (safety cap; early stopping fires first).
DATASET_EPOCHS: Dict[str, int] = {
    "ml-1m":  600,
    "Beauty": 300,
    "Steam":  250,
    "yelp":   250,
}

# Map CONGA dataset names → BSARec/WEARec data_name (matches .txt filename).
DATASET_BSAREC_NAME: Dict[str, str] = {
    "Beauty": "Beauty",
    "ml-1m":  "ML-1M",
    "Steam":  "Steam",
    "yelp":   "Yelp",
}

# Directory where CONGA bench runs store their args.txt (for show_configs.py).
CONGA_CFG_DIRS: Dict[str, str] = {
    "Beauty": "Beauty_bench_conga",
    "ml-1m":  "ml-1m_bench_conga",
    "Steam":  "Steam_bench_conga",
    "yelp":   "yelp_bench_conga",
}

# Shared output directory for BSARec-format data files exported from CONGA bins.
BSAREC_DATA_DIR: Path = SRC_DIR / "data_bsarec"

# Paths to reference main.py scripts
BSAREC_MAIN = REFS_DIR / "BSARec" / "src" / "main.py"
WEAREC_MAIN  = REFS_DIR / "WEARec" / "src" / "main.py"

# ---------------------------------------------------------------------------
# Best hyperparameter configs in BSARec / WEARec native argument names.
#
# BSARec (AAAI 2024) best configs taken from the paper's README examples and
# Table 3:
#   Beauty: lr=0.0005, alpha=0.7, c=5,  heads=1, dropout=0.5, maxlen=50
#   ML-1M:  lr=0.001,  alpha=0.7, c=15, heads=1, dropout=0.2, maxlen=200
#   Yelp:   lr=0.001,  alpha=0.9, c=5,  heads=1, dropout=0.5, maxlen=50
#   Steam:  lr=0.001,  alpha=0.7, c=5,  heads=2, dropout=0.5, maxlen=50
#
# WEARec (AAAI 2026) best configs from published output checkpoint filenames:
#   WEARec_K_50_ML-1M_0.1_0.0005_0.3_2.pt   → maxlen=50, alpha=0.1, lr=0.0005, dropout=0.3, num_heads=2
#   WEARec_K_50_Beauty_0.5_0.0005_0.2_8.pt  → maxlen=50, alpha=0.5, lr=0.0005, dropout=0.2, num_heads=8
# ---------------------------------------------------------------------------
BASELINE_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "bsarec": {
        "Beauty": {"lr": 0.0005, "alpha": 0.7, "c": 5,  "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "ml-1m":  {"lr": 0.001,  "alpha": 0.7, "c": 15, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.2, "hidden_dropout_prob": 0.2,
                   "max_seq_length": 200, "batch_size": 128},
        "Steam":  {"lr": 0.001,  "alpha": 0.7, "c": 5,  "num_attention_heads": 2,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "yelp":   {"lr": 0.001,  "alpha": 0.9, "c": 5,  "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
    },
    "sasrec": {
        "Beauty": {"lr": 0.001, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "ml-1m":  {"lr": 0.001, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.2, "hidden_dropout_prob": 0.2,
                   "max_seq_length": 200, "batch_size": 128},
        "Steam":  {"lr": 0.001, "num_attention_heads": 2,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "yelp":   {"lr": 0.001, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
    },
    "bert4rec": {
        "Beauty": {"lr": 0.001, "num_attention_heads": 1, "mask_ratio": 0.2,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "ml-1m":  {"lr": 0.001, "num_attention_heads": 1, "mask_ratio": 0.15,
                   "attention_probs_dropout_prob": 0.2, "hidden_dropout_prob": 0.2,
                   "max_seq_length": 200, "batch_size": 128},
        "Steam":  {"lr": 0.001, "num_attention_heads": 2, "mask_ratio": 0.2,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "yelp":   {"lr": 0.001, "num_attention_heads": 1, "mask_ratio": 0.2,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
    },
    "fmlprec": {
        "Beauty": {"lr": 0.001,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "ml-1m":  {"lr": 0.001,
                   "attention_probs_dropout_prob": 0.2, "hidden_dropout_prob": 0.2,
                   "max_seq_length": 200, "batch_size": 128},
        "Steam":  {"lr": 0.001,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "yelp":   {"lr": 0.001,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
    },
    "gru4rec": {
        "Beauty": {"lr": 0.001, "gru_hidden_size": 64,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "ml-1m":  {"lr": 0.001, "gru_hidden_size": 64,
                   "attention_probs_dropout_prob": 0.2, "hidden_dropout_prob": 0.2,
                   "max_seq_length": 200, "batch_size": 128},
        "Steam":  {"lr": 0.001, "gru_hidden_size": 64,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
        "yelp":   {"lr": 0.001, "gru_hidden_size": 64,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50},
    },
    "duorec": {
        "Beauty": {"lr": 0.001, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50,
                   "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
        "ml-1m":  {"lr": 0.001, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.2, "hidden_dropout_prob": 0.2,
                   "max_seq_length": 200, "batch_size": 128,
                   "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
        "Steam":  {"lr": 0.001, "num_attention_heads": 2,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50,
                   "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
        "yelp":   {"lr": 0.001, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50,
                   "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
    },
    "fearec": {
        "Beauty": {"lr": 0.001, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50,
                   "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot",
                   "spatial_ratio": 0.1, "global_ratio": 0.6, "fredom": "True", "fredom_type": "us_x"},
        "ml-1m":  {"lr": 0.001, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.2, "hidden_dropout_prob": 0.2,
                   "max_seq_length": 200, "batch_size": 128,
                   "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot",
                   "spatial_ratio": 0.1, "global_ratio": 0.6, "fredom": "True", "fredom_type": "us_x"},
        "Steam":  {"lr": 0.001, "num_attention_heads": 2,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50,
                   "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot",
                   "spatial_ratio": 0.1, "global_ratio": 0.6, "fredom": "True", "fredom_type": "us_x"},
        "yelp":   {"lr": 0.001, "num_attention_heads": 1,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5, "max_seq_length": 50,
                   "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot",
                   "spatial_ratio": 0.1, "global_ratio": 0.6, "fredom": "True", "fredom_type": "us_x"},
    },
    # WEARec (AAAI 2026) – num_heads is WEARec-specific (frequency filter heads),
    # num_attention_heads is the standard BSARec-style base arg (kept at default 2).
    "wearec": {
        "Beauty": {"lr": 0.0005, "num_heads": 8,
                   "attention_probs_dropout_prob": 0.2, "hidden_dropout_prob": 0.2,
                   "max_seq_length": 50, "alpha": 0.5},
        "ml-1m":  {"lr": 0.0005, "num_heads": 2,
                   "attention_probs_dropout_prob": 0.3, "hidden_dropout_prob": 0.3,
                   "max_seq_length": 50, "alpha": 0.1},
        "Steam":  {"lr": 0.001,  "num_heads": 4,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5,
                   "max_seq_length": 50, "alpha": 0.3},
        "yelp":   {"lr": 0.001,  "num_heads": 4,
                   "attention_probs_dropout_prob": 0.5, "hidden_dropout_prob": 0.5,
                   "max_seq_length": 50, "alpha": 0.3},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args_txt(path: Path) -> Dict[str, str]:
    """Read an args.txt file (comma-separated key,value per line) → dict."""
    result: Dict[str, str] = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if ',' in line:
                    k, v = line.split(',', 1)
                    result[k.strip()] = v.strip()
    except OSError:
        pass
    return result


def _cfg_to_cli(cfg: Dict[str, Any]) -> List[str]:
    """Convert a config dict to a flat CLI arg list.

    bool True  → --key (flag only)
    other      → --key value
    """
    cli: List[str] = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        else:
            cli.extend([f"--{k}", str(v)])
    return cli


def ensure_bsarec_data(dataset: str) -> None:
    """Preprocess dataset via CONGA's utils and export to BSARec .txt format.

    The exported file lives at BSAREC_DATA_DIR/{bsarec_name}.txt.
    Re-uses existing file if already present.
    """
    bsarec_name = DATASET_BSAREC_NAME.get(dataset, dataset)
    out_path = BSAREC_DATA_DIR / f"{bsarec_name}.txt"
    if out_path.exists():
        return
    print(f"[data] Preprocessing {dataset} and exporting to BSARec format → {out_path}")
    check_and_convert_dataset(dataset)
    export_bsarec_txt(dataset, out_path)


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def conga_command(dataset: str, patience: int,
                  epochs_override: Optional[int] = None) -> List[str]:
    cfg = CONGA_CONFIGS.get(dataset, CONGA_CONFIGS["ml-1m"])
    if epochs_override is not None:
        cfg = {**cfg, "num_epochs": epochs_override}
    cli = _cfg_to_cli(cfg) + ["--patience", str(patience)]
    return [
        "uv", "run", "python", "main.py",
        "--model_type", "conga",
        "--dataset", dataset,
        "--train_dir", "bench_conga",
        *cli,
    ]


def baseline_command(dataset: str, model: str, patience: int,
                     extra: List[str],
                     epochs_override: Optional[int] = None) -> List[str]:
    """Build the command that invokes BSARec's (or WEARec's) original main.py.

    Arg names match each paper's native CLI exactly:
      --max_seq_length, --hidden_size, --num_hidden_layers,
      --num_attention_heads, --attention_probs_dropout_prob,
      --hidden_dropout_prob, --epochs, --batch_size, --lr, ...
    """
    bsarec_name = DATASET_BSAREC_NAME.get(dataset, dataset)
    epochs = (epochs_override
              if epochs_override is not None
              else DATASET_EPOCHS.get(dataset, 300))

    cfg = BASELINE_CONFIGS.get(model, {}).get(dataset, {})
    out_dir = SRC_DIR / f"{dataset}_bench_{model}"

    # WEARec uses its own main.py (includes the WEARec model implementation).
    ref_main = str(WEAREC_MAIN if model == "wearec" else BSAREC_MAIN)

    return [
        "uv", "run", "python", ref_main,
        "--data_name",      bsarec_name,
        "--data_dir",       str(BSAREC_DATA_DIR) + "/",
        "--output_dir",     str(out_dir) + "/",
        "--train_name",     "log",
        "--model_type",     model,
        "--epochs",         str(epochs),
        "--patience",       str(patience),
        "--hidden_size",    "64",
        "--num_hidden_layers", "2",
        *_cfg_to_cli(cfg),
        *extra,
    ]


def run_output_dir(dataset: str, model: str) -> Path:
    return SRC_DIR / f"{dataset}_bench_{model}"


def has_results(out_dir: Path) -> bool:
    # Accept both log.txt (CONGA) and log.log (BSARec reference output).
    for name in ("log.txt", "log.log"):
        f = out_dir / name
        try:
            if f.exists() and f.stat().st_size > 0:
                return True
        except OSError:
            pass
    return False


def run_one(cmd: List[str], out_dir: Path, console_log: Path,
            dry: bool = False) -> int:
    print("\n" + "=" * 78)
    print(f">>> {out_dir.name}")
    print("CMD:", " ".join(cmd))
    print(f"STDOUT/ERR -> {console_log}")
    print("=" * 78, flush=True)
    if dry:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    console_log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with console_log.open("w") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(SRC_DIR),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0
    # BSARec writes {train_name}.log; rename to log.txt for consistency.
    log_log = out_dir / "log.log"
    log_txt = out_dir / "log.txt"
    if log_log.exists() and not log_txt.exists():
        log_log.rename(log_txt)
    print(f"<<< {out_dir.name} finished (rc={proc.returncode}) in {dt/60:.1f} min")
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   choices=DATASETS, help="Datasets to benchmark.")
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS, help="Models to run.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override num_epochs for all runs.")
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience (number of epochs).")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra CLI args forwarded to reference main.py after '--'.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if the output directory already has log.txt.")
    p.add_argument("--continue_on_error", action="store_true",
                   help="Keep going if one run fails (default stops).")
    p.add_argument("--logs_dir", default="benchmarks_logs",
                   help="Directory (relative to src/) for per-run stdout logs.")
    args = p.parse_args()

    logs_root = SRC_DIR / args.logs_dir
    logs_root.mkdir(parents=True, exist_ok=True)

    plan: List[tuple] = []
    for ds in args.datasets:
        for m in args.models:
            plan.append((ds, m))

    print(f"[plan] {len(plan)} runs:")
    for ds, m in plan:
        print(f"   - {ds} × {m}")

    # Pre-export data for all datasets that have baseline runs planned.
    baseline_datasets = {ds for ds, m in plan if m != "conga"}
    if not args.dry_run:
        for ds in sorted(baseline_datasets):
            ensure_bsarec_data(ds)

    failures: List[tuple] = []
    t_start = time.time()

    for ds, m in plan:
        out_dir = run_output_dir(ds, m)
        if not args.force and has_results(out_dir):
            print(f"[skip] {out_dir.name} already has results (use --force to override)")
            continue

        if m == "conga":
            cmd = conga_command(ds, args.patience, epochs_override=args.epochs)
        else:
            cmd = baseline_command(ds, m, args.patience, args.extra,
                                   epochs_override=args.epochs)

        console_log = logs_root / f"{ds}_{m}.out"
        rc = run_one(cmd, out_dir, console_log, dry=args.dry_run)
        if rc != 0:
            failures.append((ds, m, rc))
            print(f"[error] {ds}/{m} exit code {rc} (see {console_log})")
            if not args.continue_on_error and not args.dry_run:
                print("Aborting. Pass --continue_on_error to keep going.")
                break

    dt = (time.time() - t_start) / 60
    print("\n" + "=" * 78)
    print(f"Total elapsed: {dt:.1f} min")
    if failures:
        print("Failures:")
        for ds, m, rc in failures:
            print(f"  - {ds}/{m} (rc={rc})")
    else:
        print("All runs completed successfully.")
    print("=" * 78)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

