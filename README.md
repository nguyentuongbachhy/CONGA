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

CONGA is a sequential recommendation system based on SASRec with extensions for graph-based learning, pattern mining, and continual learning capabilities.

---

## Repository Structure

```
CONGA/
├── code/
│   ├── notebooks/
│   │   └── kaggle_sasrec_training.ipynb
│   ├── refs/                      # Reference implementations
│   │   ├── DuoRec/
│   │   └── SASRec.pytorch/
│   └── src/
│       ├── main.py                # Main training entry point
│       ├── models/                # Model implementations
│       │   ├── __init__.py
│       │   ├── model.py           # SASRec model
│       │   ├── graph_model.py     # Graph model
│       │   ├── graph_teacher.py   # Graph teacher (LightGCN)
│       │   ├── continuum_memory.py # Continual learning memory
│       │   └── sasrec_integration.py # Graph-SASRec integration
│       ├── components/            # Model components
│       │   ├── encoder.py
│       │   ├── ffn.py
│       │   ├── mhc.py
│       │   └── rope.py
│       ├── utils/                 # Utilities
│       │   ├── __init__.py
│       │   ├── common.py          # Common utilities
│       │   ├── evaluation.py      # Evaluation functions
│       │   ├── preprocessing.py   # Data preprocessing
│       │   └── memory.py          # Memory utilities
│       ├── training/              # Training scripts
│       │   ├── train_sasrec_with_graph.py
│       │   ├── train_with_patterns.py
│       │   └── train_teacher.py
│       ├── experiments/           # Benchmarking & tuning
│       │   ├── benchmark_configs.py
│       │   ├── run_benchmark.py
│       │   ├── run_tuning.py
│       │   └── tune_hyperparams.py
│       ├── pattern_mining/        # Pattern mining
│       │   ├── graph_pattern_miner.py
│       │   ├── mine_patterns.py
│       │   └── pattern_utils.py
│       ├── modules/               # CUDA modules (DO NOT MODIFY)
│       │   ├── mhc/
│       │   ├── rope/
│       │   └── swiglu/
│       ├── data/                  # Dataset files
│       └── bins/                  # Preprocessed data
├── papers/                        # Research papers
└── requirements.txt
```

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

Datasets should be in sequential interaction format at `code/src/data/`:

```
code/src/data/ml-1m.txt
code/src/data/Beauty.txt
code/src/data/Steam.txt
code/src/data/Video.txt
```

Format per line:

```
<user_id> <item_id>
```

The preprocessing script will convert raw data to binary format in `code/src/bins/`.

---

## Training

### Basic Training

Train SASRec model:

```bash
cd code/src
python main.py --dataset ml-1m --train_dir ml-1m_run --batch_size 128 --lr 0.001 --maxlen 200 --hidden_units 50 --num_blocks 2 --num_epochs 200 --num_heads 1 --dropout_rate 0.2 --device cuda
```

### Training with Graph Initialization

First train graph teacher model:

```bash
cd code/src
python training/train_teacher.py --dataset ml-1m --embedding_dim 50 --num_layers 3 --num_epochs 100 --device cuda
```

Then train SASRec with graph embeddings:

```bash
cd code/src
python training/train_sasrec_with_graph.py --dataset ml-1m --graph_embedding_path pretrained_embeddings/ml-1m_graph_embeddings.pt --batch_size 128 --device cuda
```

### Pattern Mining

Mine patterns first:

```bash
cd code/src
python -m pattern_mining.mine_patterns --dataset ml-1m --output_file pattern_data/ml-1m_patterns.pkl
```

Then train with patterns:

```bash
cd code/src
python training/train_with_patterns.py --dataset ml-1m --pattern_file pattern_data/ml-1m_patterns.pkl --use_pattern_init --use_pattern_reg --device cuda
```

### Hyperparameter Tuning

Run automated hyperparameter search:

```bash
cd code/src
python experiments/tune_hyperparams.py --dataset ml-1m --n_trials 50 --device cuda
```

### Benchmarking

Run benchmarks with different configurations:

```bash
cd code/src
python experiments/benchmark_configs.py --dataset ml-1m --pattern_file pattern_data/ml-1m_patterns.pkl --graph_emb_file pretrained_embeddings/ml-1m_graph_embeddings.pt
```

---

## Evaluation

Evaluate a trained checkpoint:

```bash
cd code/src
python eval.py --dataset ml-1m --checkpoint <path_to_checkpoint> --device cuda
```

Example:

```bash
cd code/src
python eval.py --dataset ml-1m --checkpoint ml-1m_run/SASRec.epoch=200.pth --hidden_units 50 --num_blocks 2 --num_heads 1 --maxlen 200 --device cuda
```

> [!IMPORTANT]
> Evaluation uses dataset-provided negative items to avoid false negatives/positives.

---

## Results

Performance on **MovieLens-1M** with corrected evaluation protocol (100 negatives per user):

| Configuration | NDCG@10 | HR@10 |
|--------------|---------|-------|
| SASRec Baseline | - | - |
| + Graph Init | - | - |
| + Pattern Mining | - | - |
| + Continual Learning | - | - |

> [!NOTE]
> Results will be updated after running experiments with the cleaned codebase.

---

## References

1. Kang & McAuley. "Self-Attentive Sequential Recommendation." ICDM 2018.
2. Xie et al. "Contrastive Learning for Sequential Recommendation." SIGIR 2021.
3. Zhang et al. "Enhancing Sequential Recommendation with Graph Contrastive Learning." IJCAI 2022.
4. Chang et al. "Sequential Recommendation with Graph Neural Networks." SIGIR 2021.

---

## License

MIT License
