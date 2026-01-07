# Loss Functions for Sequential Recommendation

This directory contains various loss functions for training sequential recommendation models.

## Available Loss Functions

### 1. ListMLE Loss (`listmle_loss.py`)

**Type:** Listwise ranking loss  
**Paper:** "Listwise approach to learning to rank: theory and algorithm" (ICML 2008)

**Description:**
- Maximizes likelihood of correct permutation using Plackett-Luce model
- Directly optimizes for ranking metrics like NDCG@K
- Complexity: O(n log n)

**Usage:**
```python
from src.losses import ListMLELoss, ListMLELossSimplified

# Full version (for general ranking)
loss_fn = ListMLELoss(reduction='mean', eps=1e-10)
loss = loss_fn(logits, labels, mask=None)

# Simplified version (for sequential rec: 1 positive + N negatives)
loss_fn = ListMLELossSimplified(reduction='mean', eps=1e-10)
loss = loss_fn(pos_logits, neg_logits, mask=None)
```

**Config:**
```yaml
loss:
  type: listmle_simple  # or 'listmle'
  reduction: mean
  eps: 1.0e-10
```

**Expected Improvement:** +1-2% NDCG@10

---

### 2. NeuralNDCG Loss (`neuralndcg_loss.py`)

**Type:** Direct NDCG optimization via differentiable sorting  
**Paper:** "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting" (SIGIR eCom 2021)

**Description:**
- Uses NeuralSort for differentiable sorting
- Direct optimization of NDCG@K metric
- Best metric alignment for NDCG evaluation
- Complexity: O(n² log n)

**Usage:**
```python
from src.losses import NeuralNDCGLoss, ApproxNDCGLoss

# NeuralNDCG (best accuracy)
loss_fn = NeuralNDCGLoss(k=10, tau=1.0, reduction='mean')
loss = loss_fn(logits, labels, mask=None)

# ApproxNDCG (faster, classical baseline)
loss_fn = ApproxNDCGLoss(k=10, alpha=10.0, reduction='mean')
loss = loss_fn(logits, labels, mask=None)
```

**Config:**
```yaml
loss:
  type: neuralndcg  # or 'approxndcg'
  k: 10  # NDCG@10
  tau: 1.0  # Temperature (lower = sharper, higher = smoother)
  reduction: mean
```

**Hyperparameters:**
- `k`: Cutoff for NDCG@K (should match evaluation metric)
- `tau`: Temperature for NeuralSort
  - Range: [0.1, 2.0]
  - Lower (0.1-0.5): Sharper, closer to hard sorting, may have gradient issues
  - Medium (0.5-1.5): Balanced
  - Higher (1.5-2.0): Smoother gradients, less accurate approximation
- `alpha`: Steepness for ApproxNDCG (only for ApproxNDCGLoss)
  - Range: [5.0, 20.0]
  - Higher = sharper approximation

**Expected Improvement:** +2-4% NDCG@10 (highest metric alignment)

---

### 3. BPR Loss (`bpr_loss.py`)

**Type:** Pairwise ranking loss  
**Paper:** "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)

**Usage:**
```python
from src.losses import BPRLoss

loss_fn = BPRLoss(reduction='mean')
loss = loss_fn(pos_logits, neg_logits, mask=None)
```

---

### 4. BCE Loss (`bce_loss.py`)

**Type:** Binary classification loss  
**Note:** Recent research (CEUR-WS 2025) shows BCE has tightest bound to NDCG on ML-1M

**Usage:**
```python
from src.losses import BCELoss

loss_fn = BCELoss(reduction='mean')
loss = loss_fn(pos_logits, neg_logits, mask=None)
```

---

## Loss Comparison

| Loss | Type | Complexity | NDCG Alignment | Expected Gain | Difficulty |
|------|------|------------|----------------|---------------|------------|
| BCE | Pointwise | O(n) | Medium | Baseline | Easy |
| BPR | Pairwise | O(n) | Medium | +1-3% | Easy |
| ListMLE | Listwise | O(n log n) | High | +1-2% | Medium |
| NeuralNDCG | Listwise | O(n² log n) | **Highest** | +2-4% | Medium-High |
| ApproxNDCG | Listwise | O(n²) | High | +1-2% | Medium |

---

## Recommended Usage

### Phase 1: Quick Test (ListMLE)
```yaml
# config/experiments/listmle_test.yaml
loss:
  type: listmle_simple
  reduction: mean
```

**Why:** Easy to implement, proven results, good foundation

### Phase 2: Best Metric Alignment (NeuralNDCG)
```yaml
# config/experiments/neuralndcg_test.yaml
loss:
  type: neuralndcg
  k: 10
  tau: 1.0
  reduction: mean
```

**Why:** Direct NDCG@10 optimization, highest expected gain

### Phase 3: Graph Distillation
Use the best performing loss from Phase 1-2 as the base loss for graph distillation experiments.

---

## Implementation Notes

### For Sequential Recommendation

Most sequential rec models output:
- `pos_logits`: [batch_size, seq_len] or [batch_size]
- `neg_logits`: [batch_size, seq_len, num_neg] or [batch_size, num_neg]

**ListMLELossSimplified** is designed for this case.

For **NeuralNDCGLoss**, you need to convert to full ranking format:
```python
# Concatenate pos and neg logits
all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
# Create binary labels (1 for pos, 0 for neg)
labels = torch.zeros_like(all_logits)
labels[:, 0] = 1.0  # First item is positive

loss = neuralndcg_loss(all_logits, labels)
```

### Memory Considerations

- **ListMLE**: O(n) memory
- **NeuralNDCG**: O(n²) memory due to permutation matrix
  - For large item catalogs, consider sampling negatives
  - Recommended: 100-500 items per batch

### Gradient Flow

- **ListMLE**: Smooth gradients, stable training
- **NeuralNDCG**: May need gradient clipping with low tau
  - Recommended: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

---

## References

1. **ListMLE**: Xia et al. "Listwise approach to learning to rank: theory and algorithm" (ICML 2008)
2. **NeuralNDCG**: Pobrotyn et al. "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting" (SIGIR eCom 2021)
3. **ApproxNDCG**: Qin et al. "A General Approximation Framework for Direct Optimization of Information Retrieval Measures" (ICML 2008)
4. **BCE for RecSys**: Di Teodoro et al. "Comparing Recommendation Losses under Negative Sampling" (CEUR-WS 2025)
