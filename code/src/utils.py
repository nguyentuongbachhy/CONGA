import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# MANS: Manifold-Aware Negative Sampling
# =============================================================================
# Novel negative sampling strategies that go beyond random sampling:
# - random: Standard uniform random sampling (baseline)
# - popularity: Sample popular items as harder negatives
# - frequency: Inverse frequency weighted (rare items as negatives)
# - mans: Full MANS combining popularity (50%) + frequency (30%) + random (20%)
# =============================================================================


def compute_item_popularity(dataset_name: str, itemnum: int) -> np.ndarray:
    """
    Pre-compute item popularity distribution from training data.

    Returns:
        popularity: Array of shape [itemnum+1] with interaction counts per item
    """
    popularity = np.zeros(itemnum + 1, dtype=np.float32)

    data_path = Path(f"data/{dataset_name}.txt")
    if data_path.exists():
        with open(data_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    item_id = int(parts[1])
                    if 1 <= item_id <= itemnum:
                        popularity[item_id] += 1

    # Add small epsilon to avoid zero probabilities
    popularity[1:] = popularity[1:] + 1e-6
    popularity[0] = 0  # Padding item has zero probability

    return popularity


def create_sampling_distribution(
    popularity: np.ndarray, mode: str = "popularity", smoothing: float = 0.75
) -> np.ndarray:
    """
    Create sampling probability distribution based on mode.

    Args:
        popularity: Raw item popularity counts
        mode: 'popularity', 'frequency', or 'uniform'
        smoothing: Smoothing exponent (0.75 is common, like word2vec)

    Returns:
        probs: Normalized probability distribution for sampling
    """
    if mode == "popularity":
        # Popular items have higher probability (harder negatives)
        # Apply smoothing to avoid extreme probabilities
        probs = np.power(popularity, smoothing)
    elif mode == "frequency":
        # Rare items have higher probability (diverse negatives)
        # Inverse frequency with smoothing
        probs = np.power(1.0 / (popularity + 1e-6), smoothing)
        probs = np.clip(probs, 0, 1e6)  # Prevent overflow
    else:  # uniform
        probs = np.ones_like(popularity)

    # Zero out padding
    probs[0] = 0

    # Normalize to probability distribution
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        # Fallback to uniform
        probs[1:] = 1.0 / (len(probs) - 1)
        probs[0] = 0

    return probs


def check_and_convert_dataset(dataset_name: str) -> None:
    bin_dir = Path(f"bins/{dataset_name}_bin")
    if not bin_dir.exists() or not (bin_dir / "all_items.npy").exists():
        print(f"Binary data not found for {dataset_name}. Running conversion...")
        from preprocess import convert_to_bin

        convert_to_bin(dataset_name)
        print("Conversion complete.")


def load_metadata(dataset_name: str) -> Tuple[int, int]:
    bin_dir = Path(f"bins/{dataset_name}_bin")
    meta_path = bin_dir / "meta.txt"
    with open(meta_path, "r") as f:
        usernum, itemnum = map(int, f.read().strip().split(","))
    return usernum, itemnum


def build_index(dataset_name: str) -> Tuple[List[List[int]], List[List[int]]]:
    ui_mat = np.loadtxt(f"data/{dataset_name}.txt", dtype=np.int32)
    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]
    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])
    return u2i_index, i2u_index


def data_partition(fname: str) -> Tuple[Dict, Dict, Dict, int, int]:
    from collections import defaultdict

    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    with open(f"data/{fname}.txt", "r") as f:
        for line in f:
            u, i = line.rstrip().split(" ")
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return user_train, user_valid, user_test, usernum, itemnum


class SASRecDataset(Dataset):
    """
    SASRec Dataset with MANS (Manifold-Aware Negative Sampling) support.

    Negative sampling modes:
    - 'random': Standard uniform random sampling (baseline)
    - 'popularity': Sample popular items as harder negatives (50% weight)
    - 'frequency': Inverse frequency weighted - rare items as negatives
    - 'mans': Full MANS combining popularity (50%) + frequency (30%) + random (20%)
    """

    # Class-level cache for popularity distributions (shared across instances)
    _popularity_cache: Dict[str, np.ndarray] = {}
    _pop_dist_cache: Dict[str, np.ndarray] = {}
    _freq_dist_cache: Dict[str, np.ndarray] = {}

    def __init__(
        self,
        dataset_name: str,
        maxlen: int,
        mode: str = "train",
        num_negatives: int = 1,
        neg_sampling_mode: str = "random",
    ):
        self.dataset_name = dataset_name
        self.maxlen = maxlen
        self.mode = mode
        self.num_negatives = max(1, int(num_negatives))
        self.neg_sampling_mode = neg_sampling_mode

        bin_dir = Path(f"bins/{dataset_name}_bin")
        self.all_items = np.load(bin_dir / "all_items.npy", mmap_mode="r")
        self.user_index = np.load(bin_dir / "user_index.npy", mmap_mode="r")
        self.usernum, self.itemnum = load_metadata(dataset_name)

        self.valid_users = np.where(self.user_index[:, 1] > 0)[0]
        if mode == "train":
            self.valid_users = self.valid_users[
                self.user_index[self.valid_users, 1] >= 4
            ]

        # Initialize MANS distributions if needed (only for training)
        if mode == "train" and neg_sampling_mode != "random":
            self._init_mans_distributions()

    def _init_mans_distributions(self):
        """Initialize popularity and frequency distributions for MANS."""
        cache_key = self.dataset_name

        # Compute popularity if not cached
        if cache_key not in SASRecDataset._popularity_cache:
            print(f"  [MANS] Computing item popularity distribution...")
            popularity = compute_item_popularity(self.dataset_name, self.itemnum)
            SASRecDataset._popularity_cache[cache_key] = popularity

            # Pre-compute sampling distributions
            SASRecDataset._pop_dist_cache[cache_key] = create_sampling_distribution(
                popularity, mode="popularity", smoothing=0.75
            )
            SASRecDataset._freq_dist_cache[cache_key] = create_sampling_distribution(
                popularity, mode="frequency", smoothing=0.5
            )
            print(f"  [MANS] Distributions ready. Mode: {self.neg_sampling_mode}")

        # Store references for fast access
        self.popularity = SASRecDataset._popularity_cache[cache_key]
        self.pop_dist = SASRecDataset._pop_dist_cache[cache_key]
        self.freq_dist = SASRecDataset._freq_dist_cache[cache_key]

        # Pre-compute item indices for sampling (exclude padding idx 0)
        self.item_indices = np.arange(1, self.itemnum + 1, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.valid_users)

    def _get_user_sequence(self, uid: int) -> np.ndarray:
        offset, length = self.user_index[uid]
        return np.array(self.all_items[offset : offset + length], dtype=np.int32)

    def _sample_negatives_random(
        self, num_samples: int, exclude_items: np.ndarray
    ) -> np.ndarray:
        """Standard random negative sampling."""
        candidates = np.random.randint(
            1, self.itemnum + 1, size=num_samples * 2, dtype=np.int32
        )
        mask = np.isin(candidates, exclude_items, invert=True)
        neg_flat = candidates[mask]

        while len(neg_flat) < num_samples:
            n_needed = num_samples - len(neg_flat)
            extra_cand = np.random.randint(
                1, self.itemnum + 1, size=n_needed * 2, dtype=np.int32
            )
            extra_mask = np.isin(extra_cand, exclude_items, invert=True)
            neg_flat = np.concatenate((neg_flat, extra_cand[extra_mask]))

        return neg_flat[:num_samples]

    def _sample_negatives_weighted(
        self, num_samples: int, exclude_items: np.ndarray, distribution: np.ndarray
    ) -> np.ndarray:
        """Weighted negative sampling using pre-computed distribution."""
        # Create a mask for valid items (exclude user's items)
        valid_mask = np.ones(len(distribution), dtype=np.float32)
        valid_mask[exclude_items] = 0
        valid_mask[0] = 0  # Exclude padding

        # Adjust distribution
        adj_dist = distribution * valid_mask
        total = adj_dist.sum()
        if total > 0:
            adj_dist = adj_dist / total
        else:
            # Fallback to uniform
            adj_dist[1:] = valid_mask[1:] / valid_mask[1:].sum()

        # Sample from adjusted distribution
        try:
            neg_flat = np.random.choice(
                len(adj_dist), size=num_samples, replace=True, p=adj_dist
            ).astype(np.int32)
        except ValueError:
            # Fallback to random if distribution is invalid
            neg_flat = self._sample_negatives_random(num_samples, exclude_items)

        return neg_flat

    def _sample_negatives_mans(
        self, num_samples: int, exclude_items: np.ndarray
    ) -> np.ndarray:
        """
        MANS: Manifold-Aware Negative Sampling
        Combines: popularity (50%) + frequency (30%) + random (20%)
        """
        n_pop = int(num_samples * 0.5)  # 50% popularity-weighted (harder)
        n_freq = int(num_samples * 0.3)  # 30% frequency-weighted (diverse)
        n_rand = num_samples - n_pop - n_freq  # 20% random (exploration)

        neg_parts = []

        # Popularity-weighted negatives (harder negatives)
        if n_pop > 0:
            pop_negs = self._sample_negatives_weighted(
                n_pop, exclude_items, self.pop_dist
            )
            neg_parts.append(pop_negs)

        # Frequency-weighted negatives (rare/diverse items)
        if n_freq > 0:
            freq_negs = self._sample_negatives_weighted(
                n_freq, exclude_items, self.freq_dist
            )
            neg_parts.append(freq_negs)

        # Random negatives (exploration)
        if n_rand > 0:
            rand_negs = self._sample_negatives_random(n_rand, exclude_items)
            neg_parts.append(rand_negs)

        # Combine and shuffle
        neg_flat = np.concatenate(neg_parts)
        np.random.shuffle(neg_flat)

        return neg_flat

    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        uid = self.valid_users[idx]
        full_seq = self._get_user_sequence(uid)

        if self.mode == "train":
            seq_items = full_seq[:-2] if len(full_seq) >= 4 else full_seq
        elif self.mode == "valid":
            seq_items = full_seq[:-1]
        else:
            seq_items = full_seq

        seq_len = len(seq_items)
        if seq_len <= 1:
            return (
                uid,
                np.zeros(self.maxlen, dtype=np.int32),
                np.zeros(self.maxlen, dtype=np.int32),
                np.zeros((self.maxlen, self.num_negatives), dtype=np.int32),
            )

        input_seq = seq_items[:-1]
        target_pos = seq_items[1:]

        eff_len = min(len(input_seq), self.maxlen)

        seq = np.zeros(self.maxlen, dtype=np.int32)
        pos = np.zeros(self.maxlen, dtype=np.int32)

        seq[-eff_len:] = input_seq[-eff_len:]
        pos[-eff_len:] = target_pos[-eff_len:]

        num_samples = eff_len * self.num_negatives

        # Select negative sampling strategy based on mode
        if self.neg_sampling_mode == "random":
            neg_flat = self._sample_negatives_random(num_samples, seq_items)
        elif self.neg_sampling_mode == "popularity":
            neg_flat = self._sample_negatives_weighted(
                num_samples, seq_items, self.pop_dist
            )
        elif self.neg_sampling_mode == "frequency":
            neg_flat = self._sample_negatives_weighted(
                num_samples, seq_items, self.freq_dist
            )
        elif self.neg_sampling_mode == "mans":
            neg_flat = self._sample_negatives_mans(num_samples, seq_items)
        else:
            # Fallback to random
            neg_flat = self._sample_negatives_random(num_samples, seq_items)

        neg_data = neg_flat.reshape(eff_len, self.num_negatives)

        neg = np.zeros((self.maxlen, self.num_negatives), dtype=np.int32)
        neg[-eff_len:, :] = neg_data

        return uid, seq, pos, neg


def get_dataloader(
    dataset_name,
    maxlen,
    batch_size,
    mode="train",
    num_workers=4,
    num_negatives: int = 1,
    neg_sampling_mode: str = "random",
):
    """
    Create DataLoader with optional MANS (Manifold-Aware Negative Sampling).

    Args:
        dataset_name: Name of the dataset
        maxlen: Maximum sequence length
        batch_size: Batch size
        mode: 'train', 'valid', or 'test'
        num_workers: Number of data loading workers
        num_negatives: Number of negative samples per position
        neg_sampling_mode: Negative sampling strategy:
            - 'random': Standard uniform random sampling (default)
            - 'popularity': Sample popular items as harder negatives
            - 'frequency': Inverse frequency weighted (rare items)
            - 'mans': Full MANS combining popularity + frequency + random

    Returns:
        DataLoader instance
    """
    dataset = SASRecDataset(
        dataset_name,
        maxlen,
        mode,
        num_negatives=num_negatives,
        neg_sampling_mode=neg_sampling_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(True if num_workers > 0 else False),
    )


def _batch_evaluate_logic(model, dataset, args, mode="test"):
    train, valid, test, usernum, itemnum = dataset
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = list(range(1, usernum + 1))
    eval_batch_size = 100

    for start_idx in range(0, len(users), eval_batch_size):
        end_idx = min(start_idx + eval_batch_size, len(users))
        batch_users = users[start_idx:end_idx]
        batch_u_ids = []
        batch_seqs = []
        batch_item_indices = []
        for i, u in enumerate(batch_users):
            if (
                len(train[u]) < 1
                or (mode == "test" and len(test[u]) < 1)
                or (mode == "valid" and len(valid[u]) < 1)
            ):
                continue
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            if mode == "test":
                seq[idx] = valid[u][0]
                idx -= 1
                source_seq = train[u]
            else:
                source_seq = train[u]
            for item in reversed(source_seq):
                seq[idx] = item
                idx -= 1
                if idx == -1:
                    break
            rated = set(train[u])
            rated.add(0)
            if mode == "test":
                target_item = test[u][0]
            else:
                target_item = valid[u][0]
            item_idx = [target_item]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            batch_u_ids.append(u)
            batch_seqs.append(seq)
            batch_item_indices.append(item_idx)
        if len(batch_u_ids) == 0:
            continue
        np_u_ids = np.array(batch_u_ids)
        np_seqs = np.array(batch_seqs)
        np_items = np.array(batch_item_indices)
        predictions = model.predict(np_u_ids, np_seqs, np_items)

        target_scores = predictions[:, 0]
        ranks = (predictions > target_scores.unsqueeze(1)).sum(dim=1).float()
        ranks = ranks.cpu().numpy().astype(np.int32)
        valid_user += len(ranks)
        hits = (ranks < 10).astype(np.float32)
        ndcgs = hits * (1.0 / np.log2(ranks + 2.0))
        HT += hits.sum()
        NDCG += ndcgs.sum()
    return NDCG / valid_user, HT / valid_user


def evaluate(model, dataset: Tuple, args) -> Tuple[float, float]:
    return _batch_evaluate_logic(model, dataset, args, mode="test")


def evaluate_valid(model, dataset: Tuple, args) -> Tuple[float, float]:
    return _batch_evaluate_logic(model, dataset, args, mode="valid")
