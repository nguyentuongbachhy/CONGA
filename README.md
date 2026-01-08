# CONGA - Contrastive Nested Graph Architecture for Sequential Recommendation

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

Implementation and experiments for **CONGA** (and its enhanced variants) on sequential recommendation tasks.

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)
- [License](#license)

---

## Overview

CONGA combines:

- **Nested graph encoder** (local + global)
- **Multi-scale contrastive learning** (sequence-level + graph-level)
- Optional continual-learning components (memory / replay / EWC / nested experts)

> [!IMPORTANT]
> **Evaluation protocol has been corrected** to avoid sampling negative items that may belong to a user's interaction history.
> The dataset now provides `neg_items` for evaluation (default: 100 negatives per user), and evaluation scripts prefer these negatives.

---

## Models

| Model | Description | Entry |
|------|-------------|-------|
| `sasrec` | Self-attentive sequential recommender | `src/models/sasrec.py` |
| `cl4srec` | Contrastive sequential recommendation | `src/models/cl4srec.py` |
| `gcl4sr` | Graph contrastive sequential recommendation | `src/models/gcl4sr.py` |
| `conga` | CONGA v1 (nested graph + contrastive) | `src/models/conga.py` |
| `conga_v2` | CONGA v2 (enhanced, optional continual learning) | `src/models/conga_v2.py` |

---

## Repository Structure

```
CONGA/
├── configs/
│   ├── sasrec.yaml
│   ├── cl4srec.yaml
│   ├── gcl4sr.yaml
│   ├── conga.yaml
│   ├── conga_v2.yaml
│   └── conga_v2_fixed.yaml
├── data/
│   └── ml-1m.txt
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── lazy_dataset.py
│   ├── models/
│   ├── trainers/
│   └── utils/
└── tests/
```

> [!NOTE]
> Folder layout in the repository may include additional experimental code and third-party baselines.
> The commands in this README assume you run from the repo root.

---

## Setup

### 1) Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Preparation

This repo expects sequential interaction data at:

```
data/ml-1m.txt
```

Format per line:

```
<user_id> <item_id>
```

---

## Training

Train a model using a YAML config:

```bash
python scripts/train.py --config configs/conga.yaml --dataset ml-1m
```

Example (CPU + small batch):

```bash
python scripts/train.py --config configs/conga_v2_fixed.yaml --dataset ml-1m --device cpu --batch_size 32 --epochs 100
```

> [!NOTE]
> When running on CPU, AMP is automatically disabled in the trainer.

---

## Evaluation

Evaluate a saved checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint experiments/checkpoints/conga_v2_fixed/best_model.pt \
  --dataset ml-1m \
  --model conga_v2 \
  --device cpu \
  --max_seq_len 50 \
  --hidden_size 64 \
  --num_layers 2 \
  --num_heads 2 \
  --dropout_rate 0.3
```

> [!IMPORTANT]
> `scripts/evaluate.py` prefers dataset-provided `neg_items` during evaluation.
> This avoids false negatives/positives introduced by random sampling.

---

## Results

Evaluated on **MovieLens-1M** (`ml-1m`) with corrected evaluation protocol (100 negatives per user, excluding user history):

| Model | Checkpoint | NDCG@10 | HR@10 |
|------|------------|---------|-------|
| `conga` | `experiments/checkpoints/conga/best_model.pt` | **0.5313** | **0.7690** |
| `conga_v2` (`conga_v2_fixed`) | `experiments/checkpoints/conga_v2_fixed/best_model.pt` | **0.5369** | **0.7714** |

> [!NOTE]
> These numbers are from `scripts/evaluate.py` with evaluation negatives provided by the dataset.

---

## References

1. Kang & McAuley. "Self-Attentive Sequential Recommendation." ICDM 2018.
2. Xie et al. "Contrastive Learning for Sequential Recommendation." SIGIR 2021.
3. Zhang et al. "Enhancing Sequential Recommendation with Graph Contrastive Learning." IJCAI 2022.
4. Chang et al. "Sequential Recommendation with Graph Neural Networks." SIGIR 2021.

---

## License

MIT License
