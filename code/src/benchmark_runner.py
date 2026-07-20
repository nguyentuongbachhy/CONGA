"""
Benchmark runner for BSARec-style baseline models (bsarec, sasrec, bert4rec,
caser, gru4rec, fmlprec, duorec, fearec).

Keeps CONGA's training pipeline untouched; call `run_benchmark(args)` from
`main.py` whenever `args.model_type` is NOT "conga".

Design:
- Reuses CONGA's data_partition / bin preprocessing.
- Builds a BSARec-style `RecDataset` with prefix augmentation so each user
  contributes many (context -> next-item) samples per epoch.
- Wraps the raw benchmark model with `BenchmarkAdapter` so it exposes the
  same `predict(user_ids, seqs, item_indices)` interface CONGA's evaluator
  expects; training uses the model's native `calculate_loss`.
- Uses Adam + early stopping on val NDCG@10, mirroring BSARec/trainers.py.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from benchmark_models import MODEL_DICT
from utils import (
    check_and_convert_dataset,
    load_metadata,
    data_partition,
    evaluate,
    evaluate_valid,
)


# -------------------------------------------------------------------------
# Best hyperparameter defaults per (model, dataset). Values not listed here
# fall back to the CLI defaults (which themselves mirror BSARec/src/utils.py).
#
# ml-1m entries are aligned with CONGA ablation A4 for fair architecture comparison:
#   maxlen=300, batch_size=128, num_heads=1, dropout_rate=0.2, lr=0.001, loss_type=ce
# This ensures the ONLY variable between CONGA and baselines is model architecture.
# -------------------------------------------------------------------------
BEST_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "bsarec": {
        "Beauty":  {"lr": 0.0005, "alpha": 0.7, "c": 5,  "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50,  "loss_type": "ce"},
        "ml-1m":   {"lr": 0.001,  "alpha": 0.7, "c": 15, "num_heads": 1, "dropout_rate": 0.2, "maxlen": 300, "batch_size": 128, "loss_type": "ce"},
        "yelp":    {"lr": 0.001,  "alpha": 0.9, "c": 5,  "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50,  "loss_type": "ce"},
        "Steam":   {"lr": 0.001,  "alpha": 0.7, "c": 5,  "num_heads": 2, "dropout_rate": 0.5, "maxlen": 50,  "loss_type": "ce"},
    },
    "sasrec": {
        "Beauty":  {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50},
        "ml-1m":   {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.2, "maxlen": 300, "batch_size": 128, "loss_type": "ce"},
        "Steam":   {"lr": 0.001, "num_heads": 2, "dropout_rate": 0.5, "maxlen": 50},
        "yelp":    {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50},
    },
    "bert4rec": {
        "Beauty":  {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50,  "mask_ratio": 0.2},
        "ml-1m":   {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.2, "maxlen": 300, "batch_size": 128, "mask_ratio": 0.15},
        "Steam":   {"lr": 0.001, "num_heads": 2, "dropout_rate": 0.5, "maxlen": 50,  "mask_ratio": 0.2},
        "yelp":    {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50,  "mask_ratio": 0.2},
    },
    "fmlprec": {
        "Beauty":  {"lr": 0.001, "dropout_rate": 0.5, "maxlen": 50},
        "ml-1m":   {"lr": 0.001, "dropout_rate": 0.2, "maxlen": 300, "batch_size": 128, "loss_type": "ce"},
        "Steam":   {"lr": 0.001, "dropout_rate": 0.5, "maxlen": 50},
        "yelp":    {"lr": 0.001, "dropout_rate": 0.5, "maxlen": 50},
    },
    "wearec": {
        "Beauty":  {"lr": 0.0005, "num_heads": 8, "dropout_rate": 0.5, "maxlen": 50,  "alpha": 0.2, "loss_type": "ce"},
        "ml-1m":   {"lr": 0.001,  "num_heads": 1, "dropout_rate": 0.1, "maxlen": 300, "batch_size": 128, "alpha": 0.9, "loss_type": "ce"},
        "Steam":   {"lr": 0.001,  "num_heads": 2, "dropout_rate": 0.2, "maxlen": 50,  "alpha": 0.5, "loss_type": "ce"},
        "yelp":    {"lr": 0.001,  "num_heads": 2, "dropout_rate": 0.2, "maxlen": 50,  "alpha": 0.5, "loss_type": "ce"},
    },
    "gru4rec": {
        "Beauty":  {"lr": 0.001, "dropout_rate": 0.5, "maxlen": 50,  "gru_hidden_size": 64},
        "ml-1m":   {"lr": 0.001, "dropout_rate": 0.2, "maxlen": 300, "batch_size": 128, "gru_hidden_size": 64, "loss_type": "ce"},
        "Steam":   {"lr": 0.001, "dropout_rate": 0.5, "maxlen": 50,  "gru_hidden_size": 64},
        "yelp":    {"lr": 0.001, "dropout_rate": 0.5, "maxlen": 50,  "gru_hidden_size": 64},
    },
    "duorec": {
        "Beauty":  {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50,
                     "loss_type": "ce", "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
        "ml-1m":   {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.2, "maxlen": 300, "batch_size": 128,
                     "loss_type": "ce", "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
        "Steam":   {"lr": 0.001, "num_heads": 2, "dropout_rate": 0.5, "maxlen": 50,
                     "loss_type": "ce", "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
        "yelp":    {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50,
                     "loss_type": "ce", "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot"},
    },
    "fearec": {
        "Beauty":  {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50,
                     "loss_type": "ce", "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot",
                     "spatial_ratio": 0.1, "global_ratio": 0.6,
                     "fredom": "True", "fredom_type": "us_x"},
        "ml-1m":   {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.2, "maxlen": 300, "batch_size": 128,
                     "loss_type": "ce", "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot",
                     "spatial_ratio": 0.1, "global_ratio": 0.6,
                     "fredom": "True", "fredom_type": "us_x"},
        "Steam":   {"lr": 0.001, "num_heads": 2, "dropout_rate": 0.5, "maxlen": 50,
                     "loss_type": "ce", "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot",
                     "spatial_ratio": 0.1, "global_ratio": 0.6,
                     "fredom": "True", "fredom_type": "us_x"},
        "yelp":    {"lr": 0.001, "num_heads": 1, "dropout_rate": 0.5, "maxlen": 50,
                     "loss_type": "ce", "tau": 1.0, "lmd": 0.1, "lmd_sem": 0.1, "ssl": "us_x", "sim": "dot",
                     "spatial_ratio": 0.1, "global_ratio": 0.6,
                     "fredom": "True", "fredom_type": "us_x"},
    },
}

_CONTRASTIVE = {"duorec", "fearec"}

# Models that can supervise all positions in one forward pass (SASRec paper
# protocol). Keeping these seq-level collapses O(num_users * maxlen) prefix
# samples per epoch down to O(num_users), giving a `maxlen`x speedup while
# preserving the same number of (input, target) pairs seen per epoch.
# -------------------------------------------------------------------------
# CONGA best configs per dataset — used by run_all_benchmarks.py to build
# the conga CLI.  Bool values (True) expand to --flag; others to --key value.
# -------------------------------------------------------------------------

_CONGA_ARCH: Dict[str, Any] = {
    "hidden_units": 64, "num_blocks": 2, "loss_type": "ce",
    "num_negatives": 1,
    "cosine_anneal": True, "warmup_ratio": 0.05,
    "use_nested_learning": True, "mem_start_epoch": 9999,
    "titans_d_mem": 128, "titans_base_lr_scale": 0.1,
    "titans_mem_lr_scale": 0.5, "titans_mem_wd": 0.01,
    "eval_every": 5,
}
CONGA_CONFIGS: Dict[str, Dict[str, Any]] = {
    # num_epochs is a safety cap — auto_phase stops via patience, not this limit.
    # Configs validated via run_ablation.sh (norm_first not used in ablation runs).
    "ml-1m":  {**_CONGA_ARCH, "lr": 0.001,   "num_heads": 2, "dropout_rate": 0.2,
               "maxlen": 300, "batch_size": 128,  "mem_maxlen": 600, "num_epochs": 600},
    "Beauty": {**_CONGA_ARCH, "lr": 0.0005,  "num_heads": 2, "dropout_rate": 0.5,
               "maxlen": 50,  "batch_size": 1024, "mem_maxlen": 100, "num_epochs": 400},
    "Steam":  {**_CONGA_ARCH, "lr": 0.001,   "num_heads": 2, "dropout_rate": 0.5,
               "maxlen": 50,  "batch_size": 256,  "mem_maxlen": 100, "num_epochs": 250},
    "yelp":   {**_CONGA_ARCH, "lr": 0.001,   "num_heads": 1, "dropout_rate": 0.5,
               "maxlen": 50,  "batch_size": 256,  "mem_maxlen": 100, "num_epochs": 250},
}

SEQ_LEVEL_MODELS = {"sasrec", "bsarec", "bsarec_rope", "fmlprec", "gru4rec", "duorec",
                    "fearec", "bert4rec", "wearec"}

PREFIX_SUBSAMPLE_DEFAULT = 4


# -------------------------------------------------------------------------
# Dataset: BSARec-style prefix augmentation on CONGA's pre-processed data.
# -------------------------------------------------------------------------
class BSARecRecDataset(Dataset):
    """Feeds BSARec baselines from CONGA's `data_partition` (bin-cached
    k-core sequences).

    Supports two training modes:

    * ``train_mode="seq"`` (one sample / user, SASRec paper protocol):
      emits ``(uid, input_seq[maxlen], pos_seq[maxlen], neg_seq[maxlen],
      same_target[maxlen or 0])`` where every non-padding position supplies
      a next-item target. This mirrors CONGA's `SASRecDataset` and is the
      fast path.
    * ``train_mode="prefix"`` (original BSARec protocol): materialises
      every prefix of the user's sequence and supervises only the last
      token. Slower (`maxlen`x more forward passes) but matches BSARec
      reference code. Use ``prefix_subsample`` to keep only K random
      prefixes per user per epoch; `None` keeps every prefix.

    Validation / test modes are unaffected: they use the standard
    leave-one-out prefix emitted in the ``prefix`` format.
    """

    def __init__(
        self,
        user_train: Dict[int, List[int]],
        user_valid: Dict[int, List[int]],
        user_test: Dict[int, List[int]],
        itemnum: int,
        maxlen: int,
        mode: str = "train",
        contrastive: bool = False,
        train_mode: str = "prefix",
        prefix_subsample: Optional[int] = None,
    ) -> None:
        if train_mode not in ("seq", "prefix"):
            raise ValueError(f"Unknown train_mode: {train_mode}")
        self.maxlen = maxlen
        self.mode = mode
        self.contrastive = contrastive and mode == "train"
        self.item_size = itemnum + 1  # 0 is padding; ids in [1..itemnum]
        self.train_mode = train_mode if mode == "train" else "prefix"
        self.prefix_subsample = prefix_subsample

        # In "seq" mode we store the truncated user sequence once and build
        # per-position targets on the fly. In "prefix" mode we materialise
        # every prefix like BSARec's reference code.
        self.samples: List[Tuple[int, List[int]]] = []  # (user_id, sequence)
        self.user_ids: List[int] = []  # parallel list for contrastive lookups

        # In "prefix_subsample" mode we store the full context per user and
        # draw K fresh random prefixes on every __getitem__ call so the
        # model sees different cuts every epoch without materialising all of
        # them upfront.
        self._user_contexts: List[Tuple[int, List[int]]] = []

        if mode == "train":
            if self.train_mode == "seq":
                for uid, train_seq in user_train.items():
                    if len(train_seq) < 2:
                        continue
                    # Keep the last maxlen+1 items so we can derive input /
                    # next-item pairs without padding loss.
                    ctx = list(train_seq[-(maxlen + 1):])
                    self.samples.append((uid, ctx))
                    self.user_ids.append(uid)
            elif self.prefix_subsample is not None:
                # One logical sample per (user, slot) so DataLoader sees K
                # mini-samples per user; the actual prefix length is drawn
                # at __getitem__ time for epoch-level randomness.
                for uid, train_seq in user_train.items():
                    if len(train_seq) < 2:
                        continue
                    ctx = list(train_seq[-(maxlen + 2):])
                    self._user_contexts.append((uid, ctx))
                    for _ in range(self.prefix_subsample):
                        # samples[i] carries uid; actual prefix built lazily.
                        idx_in_ctx = len(self._user_contexts) - 1
                        self.samples.append((uid, [idx_in_ctx]))  # marker
                        self.user_ids.append(uid)
            else:
                # Full BSARec protocol: materialise every prefix.
                for uid, train_seq in user_train.items():
                    if len(train_seq) < 2:
                        continue
                    # Match BSARec: seq[-(maxlen + 2):-2] of the RAW user seq.
                    # user_train already excludes val/test tails so just cap.
                    ctx = list(train_seq[-(maxlen + 2):])
                    for i in range(1, len(ctx)):
                        self.samples.append((uid, ctx[: i + 1]))
                        self.user_ids.append(uid)
        elif mode == "valid":
            for uid, seq in user_train.items():
                v = user_valid.get(uid, [])
                if not v or not seq:
                    continue
                full = list(seq) + [v[0]]
                self.samples.append((uid, full))
                self.user_ids.append(uid)
        else:  # test
            for uid, seq in user_train.items():
                t = user_test.get(uid, [])
                v = user_valid.get(uid, [])
                if not t or not seq:
                    continue
                full = list(seq) + list(v) + [t[0]]
                self.samples.append((uid, full))
                self.user_ids.append(uid)

        # Build same-target index for DuoRec/FEARec supervised NCE. In seq
        # mode the "last item" is ctx[-1]; in prefix mode it is items[-1].
        # Never used together with prefix_subsample (Caser is the only
        # subsampled model and it is not contrastive).
        self.same_target_index: Optional[Dict[int, List[List[int]]]] = None
        if self.contrastive and self.prefix_subsample is None:
            idx: Dict[int, List[List[int]]] = {}
            for _, items in self.samples:
                idx.setdefault(items[-1], []).append(items)
            self.same_target_index = idx

    def __len__(self) -> int:
        return len(self.samples)

    def _neg_sample(self, seen: set) -> int:
        item = random.randint(1, self.item_size - 1)
        while item in seen:
            item = random.randint(1, self.item_size - 1)
        return item

    def _pad(self, seq: List[int]) -> List[int]:
        pad_len = self.maxlen - len(seq)
        if pad_len > 0:
            seq = [0] * pad_len + list(seq)
        return list(seq[-self.maxlen:])

    def _getitem_seq(self, index: int) -> Tuple[torch.Tensor, ...]:
        uid, ctx = self.samples[index]
        # ctx has up to maxlen+1 items. Input is ctx[:-1], per-position target
        # is ctx[1:]. Pad-left to maxlen.
        input_items = list(ctx[:-1])
        target_items = list(ctx[1:])
        eff = min(len(input_items), self.maxlen)

        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros(self.maxlen, dtype=np.int64)
        neg = np.zeros(self.maxlen, dtype=np.int64)

        seq[-eff:] = input_items[-eff:]
        pos[-eff:] = target_items[-eff:]

        seen = set(ctx)
        for t in range(self.maxlen - eff, self.maxlen):
            neg[t] = self._neg_sample(seen)

        uid_t = torch.from_numpy(np.asarray(uid, dtype=np.int64))
        input_t = torch.from_numpy(seq)
        pos_t = torch.from_numpy(pos)
        neg_t = torch.from_numpy(neg)

        if self.contrastive and self.same_target_index is not None:
            last_item = ctx[-1]
            bucket = self.same_target_index.get(last_item, [ctx])
            sem_aug = random.choice(bucket)
            tries = 0
            while len(bucket) > 1 and sem_aug is ctx and tries < 5:
                sem_aug = random.choice(bucket)
                tries += 1
            # sem_aug is another user's ctx (len up to maxlen+1). The
            # contrastive head only looks at the last hidden state, so we pad
            # the whole augmented sequence (drop its trailing target item to
            # mirror the encoder's input view).
            sem_input = list(sem_aug[:-1])
            sem_ctx = self._pad(sem_input)
            same_t = torch.tensor(sem_ctx, dtype=torch.long)
        else:
            same_t = torch.zeros(0, dtype=torch.long)

        return uid_t, input_t, pos_t, neg_t, same_t

    def _getitem_prefix(self, index: int) -> Tuple[torch.Tensor, ...]:
        uid, items = self.samples[index]

        # In prefix_subsample mode, `items` is a marker [ctx_idx]; draw a
        # fresh random prefix from the user's full context on every call so
        # each epoch sees different cuts.
        if (self.mode == "train" and self.prefix_subsample is not None
                and self._user_contexts):
            ctx_idx = items[0]
            _, ctx = self._user_contexts[ctx_idx]
            # Random prefix length in [2, len(ctx)] -> slice ctx[:i+1].
            i = random.randint(1, len(ctx) - 1)
            items = ctx[: i + 1]

        input_ids = items[:-1]
        answer = items[-1]

        seen = set(items)
        neg = self._neg_sample(seen)
        padded = self._pad(input_ids)

        uid_t = torch.tensor(uid, dtype=torch.long)
        input_t = torch.tensor(padded, dtype=torch.long)
        ans_t = torch.tensor(answer, dtype=torch.long)
        neg_t = torch.tensor(neg, dtype=torch.long)

        if self.contrastive and self.same_target_index is not None:
            bucket = self.same_target_index.get(answer, [items])
            sem_aug = random.choice(bucket)
            tries = 0
            while len(bucket) > 1 and sem_aug == items and tries < 5:
                sem_aug = random.choice(bucket)
                tries += 1
            sem_ctx = self._pad(list(sem_aug[:-1]))
            same_t = torch.tensor(sem_ctx, dtype=torch.long)
        else:
            same_t = torch.zeros(0, dtype=torch.long)

        return uid_t, input_t, ans_t, neg_t, same_t

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        if self.train_mode == "seq" and self.mode == "train":
            return self._getitem_seq(index)
        return self._getitem_prefix(index)


# -------------------------------------------------------------------------
# Adapter: expose CONGA's `predict(user_ids, seqs, item_indices)` API on top
# of BSARec models so CONGA's `evaluate()` / `evaluate_valid()` work unchanged.
# -------------------------------------------------------------------------
class BenchmarkAdapter(nn.Module):
    def __init__(self, model: nn.Module, device: str, model_name: str,
                 loss_type: str = "bpr") -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.model_name = model_name.lower()
        self.loss_type = (loss_type or "bpr").lower()
        # CE is only applied for non-contrastive seq-level models.
        # Contrastive models (duorec, fearec) keep their BPR+CL native loss.
        self._use_ce = (
            self.loss_type == "ce"
            and self.model_name not in ("duorec", "fearec")
        )
        # Expose item embeddings for any downstream code that needs them.
        self.item_embeddings = model.item_embeddings

    def forward(self, *args, **kwargs):  # pragma: no cover - passthrough
        return self.model(*args, **kwargs)

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        # Full-softmax CE loss: supervise every non-padding position over the
        # entire item vocabulary. More accurate gradient signal than BPR,
        # especially for smaller item sets (Beauty/yelp <10k items).
        if self._use_ce and answers.dim() == 2:
            import torch.nn.functional as F
            seq_out = self.model.forward(input_ids, user_ids)  # (B, L, H)
            mask = (answers != 0)
            feats = seq_out[mask]                              # (N_valid, H)
            targets = answers[mask].long()                    # (N_valid,)
            logits = F.linear(feats, self.model.item_embeddings.weight)  # (N_valid, V)
            return F.cross_entropy(logits, targets)
        return self.model.calculate_loss(input_ids, answers, neg_answers, same_target, user_ids)

    def predict(self, user_ids, seqs, item_indices, **_unused) -> torch.Tensor:
        # Normalise tensor types / devices.
        if isinstance(seqs, np.ndarray):
            seqs_t = torch.from_numpy(seqs).long().to(self.device)
        elif torch.is_tensor(seqs):
            seqs_t = seqs.to(self.device).long()
        else:
            seqs_t = torch.as_tensor(seqs, dtype=torch.long, device=self.device)

        if isinstance(user_ids, (list, tuple)):
            uid_t = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
        elif isinstance(user_ids, np.ndarray):
            uid_t = torch.from_numpy(user_ids).long().to(self.device)
        elif torch.is_tensor(user_ids):
            uid_t = user_ids.to(self.device).long()
        else:
            uid_t = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)

        if self.model_name == "bert4rec":
            # BERT4Rec's predict reconstructs input by appending a [MASK] token.
            seq_out = self.model.predict(seqs_t, uid_t)
        else:
            seq_out = self.model.forward(seqs_t, uid_t)

        final = seq_out[:, -1, :]  # (B, H)

        if not torch.is_tensor(item_indices):
            item_indices_t = torch.as_tensor(item_indices, dtype=torch.long, device=self.device)
        else:
            item_indices_t = item_indices.to(self.device).long()

        if item_indices_t.dim() == 1:
            item_embs = self.model.item_embeddings(item_indices_t)  # (N, H)
            logits = final @ item_embs.t()  # (B, N)
        else:
            # (B, N) -> (B, N, H); final (B, H)
            item_embs = self.model.item_embeddings(item_indices_t)
            logits = torch.bmm(item_embs, final.unsqueeze(-1)).squeeze(-1)
        return logits


# -------------------------------------------------------------------------
# Arg plumbing: map CONGA CLI names to BSARec expected attribute names and
# inject best defaults for the chosen (model, dataset) combo unless the user
# has already overridden them on the command line.
# -------------------------------------------------------------------------
def _apply_best_defaults(args: Any, cli_overrides: set) -> None:
    model = args.model_type.lower()
    cfg = BEST_CONFIGS.get(model, {}).get(args.dataset)
    if cfg is None:
        print(f"[benchmark] No preset for ({model}, {args.dataset}); using CLI defaults.")
        return
    applied = []
    for k, v in cfg.items():
        if k in cli_overrides:
            continue  # respect explicit user choice
        setattr(args, k, v)
        applied.append(f"{k}={v}")
    if applied:
        print(f"[benchmark] Applying best preset for ({model}, {args.dataset}): {', '.join(applied)}")


def _alias_args(args: Any, usernum: int, itemnum: int) -> None:
    # CONGA name -> BSARec name
    args.hidden_size = args.hidden_units
    args.max_seq_length = args.maxlen
    args.num_hidden_layers = args.num_blocks
    args.num_attention_heads = args.num_heads
    args.hidden_dropout_prob = args.dropout_rate
    # Fall back to the CONGA dropout if the user did not set
    # attention_probs_dropout_prob explicitly.
    if getattr(args, "attention_probs_dropout_prob", None) in (None, -1.0):
        args.attention_probs_dropout_prob = args.dropout_rate
    if not getattr(args, "hidden_act", None):
        args.hidden_act = "gelu"
    if getattr(args, "initializer_range", None) in (None, 0.0):
        args.initializer_range = 0.02
    args.item_size = itemnum + 1
    args.num_users = usernum + 1


# -------------------------------------------------------------------------
# Main training loop: BSARec-style Adam + early stopping on val NDCG@10.
# -------------------------------------------------------------------------
def run_benchmark(args: Any, cli_overrides: Optional[set] = None) -> None:
    cli_overrides = cli_overrides or set()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = f"{args.dataset}_{args.train_dir}"
    os.makedirs(out_dir, exist_ok=True)

    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)

    # Inject best hyperparameters for this (model, dataset) unless the user
    # explicitly set them on the CLI.
    _apply_best_defaults(args, cli_overrides)
    _alias_args(args, usernum, itemnum)

    # Persist args snapshot alongside CONGA's log file layout.
    with open(os.path.join(out_dir, "args.txt"), "w") as fa:
        fa.write("\n".join(f"{k},{v}" for k, v in sorted(vars(args).items())))

    print(f"[benchmark] model={args.model_type} | dataset={args.dataset} "
          f"| users={usernum} | items={itemnum}")
    print(f"[benchmark] maxlen={args.maxlen} | hidden={args.hidden_units} "
          f"| heads={args.num_heads} | layers={args.num_blocks} "
          f"| dropout={args.dropout_rate} | lr={args.lr} | batch={args.batch_size}")

    torch.set_float32_matmul_precision("high")

    # ---------------- Data ----------------
    user_train, user_valid, user_test, _, _ = data_partition(args.dataset)
    model_key = args.model_type.lower()
    contrastive = model_key in _CONTRASTIVE

    # Pick training mode per model. Sequence-level (1 sample/user, supervise
    # every position) is ~maxlen x faster than BSARec's prefix augmentation
    # and matches CONGA's training protocol, so every model whose encoder
    # returns a (B, L, H) tensor uses it. Caser only emits a single vector
    # per input sequence (CNN pool -> (B, 1, H)), so we fall back to prefix
    # augmentation but subsample K prefixes/user/epoch to stay fast.
    if model_key in SEQ_LEVEL_MODELS:
        train_mode = "seq"
        prefix_subsample = None
    else:
        # Fallback for any future model that emits a single vector per sequence.
        train_mode = "prefix"
        prefix_subsample = getattr(args, "prefix_subsample", None) \
            or PREFIX_SUBSAMPLE_DEFAULT

    train_ds = BSARecRecDataset(
        user_train, user_valid, user_test,
        itemnum=itemnum, maxlen=args.maxlen,
        mode="train", contrastive=contrastive,
        train_mode=train_mode, prefix_subsample=prefix_subsample,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    subs_note = f" (prefix_subsample={prefix_subsample})" if prefix_subsample else ""
    print(f"[benchmark] train_mode={train_mode}{subs_note} | "
          f"train samples = {len(train_ds)} | batches/epoch = {len(train_loader)}")

    # Reuse CONGA's evaluator -> relies on model.predict(user_ids, seqs, all_items).
    dataset_tuple = (user_train, user_valid, user_test, usernum, itemnum)
    # Force-disable CONGA memory path in the evaluator for benchmarks.
    args._mem_active = False
    setattr(args, "_bench_eval", True)

    # ---------------- Model ----------------
    raw_model = MODEL_DICT[args.model_type.lower()](args=args).to(args.device)
    model = BenchmarkAdapter(
        raw_model, args.device, args.model_type,
        loss_type=getattr(args, "loss_type", "bpr"),
    ).to(args.device)
    print(f"[benchmark] loss_type={model.loss_type} (CE active: {model._use_ce})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[benchmark] total parameters = {n_params}")

    # ---------------- Optimizer ----------------
    betas = (getattr(args, "adam_beta1", 0.9), getattr(args, "adam_beta2", 0.999))
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=betas,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )

    # ---------------- Training loop ----------------
    log_path = os.path.join(out_dir, "log.txt")
    log_f = open(log_path, "w")
    log_f.write("epoch (val_ndcg5, val_hr5, val_ndcg10, val_hr10) "
                "(test_ndcg5, test_hr5, test_ndcg10, test_hr10)\n")

    best_val_ndcg10 = 0.0
    best_val: Optional[Tuple[float, float, float, float]] = None
    best_test: Optional[Tuple[float, float, float, float]] = None
    no_improve = 0
    patience = getattr(args, "patience", 10)
    eval_every = getattr(args, "eval_every", 1)
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", unit="batch", ncols=100)

        for batch in pbar:
            batch = tuple(t.to(args.device, non_blocking=True) for t in batch)
            user_ids, input_ids, answers, neg_answers, same_target = batch

            optim.zero_grad(set_to_none=True)
            loss = model.calculate_loss(input_ids, answers, neg_answers, same_target, user_ids)
            loss.backward()
            if getattr(args, "grad_clip", 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}", end="")

        do_eval = (epoch % eval_every == 0) or (epoch == args.num_epochs)
        if not do_eval:
            print()
            continue

        model.eval()
        with torch.no_grad():
            t_valid = evaluate_valid(model, dataset_tuple, args)
        elapsed = time.time() - t0
        print(f" | Time: {elapsed:.1f}s")
        print(f"         Valid: N@5={t_valid[0]:.4f}, H@5={t_valid[1]:.4f}, "
              f"N@10={t_valid[2]:.4f}, H@10={t_valid[3]:.4f}")

        if t_valid[2] > best_val_ndcg10:
            # Test set is only scored when validation improves: aggregator
            # picks the test metrics at the best-val epoch, so anything else
            # is wasted compute.
            with torch.no_grad():
                t_test = evaluate(model, dataset_tuple, args)
            print(f"         Test : N@5={t_test[0]:.4f}, H@5={t_test[1]:.4f}, "
                  f"N@10={t_test[2]:.4f}, H@10={t_test[3]:.4f}")
            log_f.write(f"{epoch} {t_valid} {t_test}\n")
            log_f.flush()
            best_val_ndcg10 = t_valid[2]
            best_val, best_test = t_valid, t_test
            no_improve = 0
            ckpt = os.path.join(out_dir, f"{args.model_type.lower()}.best.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"         ✓ Saved best -> {ckpt}")
        else:
            no_improve += 1
            print(f"         No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\n[benchmark] Early stopping at epoch {epoch} (patience={patience}).")
                break

    log_f.close()
    print("\n[benchmark] Training completed.")
    if best_val is not None and best_test is not None:
        print(f"Best Val  - N@5: {best_val[0]:.4f}, H@5: {best_val[1]:.4f}, "
              f"N@10: {best_val[2]:.4f}, H@10: {best_val[3]:.4f}")
        print(f"Best Test - N@5: {best_test[0]:.4f}, H@5: {best_test[1]:.4f}, "
              f"N@10: {best_test[2]:.4f}, H@10: {best_test[3]:.4f}")
