"""
Lazy-loading dataset for sequential recommendation.
Reduces RAM usage by loading data on-demand instead of keeping all in memory.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import linecache

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LazySequentialDataset(Dataset):
    """
    Memory-efficient dataset that loads user sequences on-demand.
    
    Instead of loading all user sequences into RAM, this dataset:
    1. Builds an index of user positions in the file
    2. Loads sequences on-demand during __getitem__
    3. Caches recently accessed sequences (LRU cache)
    """
    
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 50,
        mode: str = "train",
        num_negatives: int = 1,
        cache_size: int = 1000,
    ):
        """
        Args:
            data_path: Path to data file (format: user_id item_id per line)
            max_seq_len: Maximum sequence length
            mode: "train", "valid", or "test"
            num_negatives: Number of negative samples per positive
            cache_size: Number of user sequences to cache in memory
        """
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.num_negatives = num_negatives
        self.cache_size = cache_size
        
        # Build index without loading all data
        self.user_index, self.num_users, self.num_items = self._build_index()
        
        # LRU cache for recently accessed sequences
        self.cache = {}
        self.cache_order = []
        
        # Get valid users for this mode
        self.valid_users = self._get_valid_users()
    
    def _build_index(self) -> Tuple[Dict[int, List[int]], int, int]:
        """
        Build an index of line numbers for each user without loading all data.
        Returns user_index mapping user_id -> list of line numbers.
        """
        user_index = defaultdict(list)
        num_users = 0
        num_items = 0
        
        with open(self.data_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                user_id = int(parts[0])
                item_id = int(parts[1])
                
                user_index[user_id].append(line_num)
                num_users = max(num_users, user_id)
                num_items = max(num_items, item_id)
        
        return dict(user_index), num_users, num_items
    
    def _load_user_sequence(self, user_id: int) -> List[int]:
        """Load a user's sequence from disk using line cache."""
        # Check cache first
        if user_id in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(user_id)
            self.cache_order.append(user_id)
            return self.cache[user_id]
        
        # Load from disk
        line_nums = self.user_index.get(user_id, [])
        items = []
        
        for line_num in line_nums:
            line = linecache.getline(self.data_path, line_num)
            parts = line.strip().split()
            if len(parts) >= 2:
                items.append(int(parts[1]))
        
        # Add to cache
        self.cache[user_id] = items
        self.cache_order.append(user_id)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        return items
    
    def _split_sequence(self, items: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """Split sequence into train/valid/test."""
        if len(items) < 3:
            return items, [], []
        
        train = items[:-2]
        valid = [items[-2]]
        test = [items[-1]]
        
        return train, valid, test
    
    def _get_valid_users(self) -> List[int]:
        """Get users with sufficient interactions for this mode."""
        valid_users = []
        
        for user_id in self.user_index.keys():
            seq = self._load_user_sequence(user_id)
            train, valid, test = self._split_sequence(seq)
            
            if self.mode == "train":
                if len(train) >= 1:
                    valid_users.append(user_id)
            elif self.mode == "valid":
                if len(train) >= 1 and len(valid) >= 1:
                    valid_users.append(user_id)
            else:  # test
                if len(train) >= 1 and len(test) >= 1:
                    valid_users.append(user_id)
        
        return valid_users
    
    def _sample_negative(self, user_items: set) -> int:
        """Sample a negative item."""
        neg = random.randint(1, self.num_items)
        while neg in user_items:
            neg = random.randint(1, self.num_items)
        return neg
    
    def __len__(self) -> int:
        return len(self.valid_users)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id = self.valid_users[idx]
        
        # Load sequence on-demand
        items = self._load_user_sequence(user_id)
        train, valid, test = self._split_sequence(items)
        
        if self.mode == "train":
            return self._get_train_sample(train)
        elif self.mode == "valid":
            return self._get_eval_sample(train, valid)
        else:
            return self._get_eval_sample(train + valid, test)
    
    def _get_train_sample(self, items: List[int]) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        user_item_set = set(items)
        
        # Initialize sequences
        seq = np.zeros(self.max_seq_len, dtype=np.int64)
        pos = np.zeros(self.max_seq_len, dtype=np.int64)
        neg = np.zeros((self.max_seq_len, self.num_negatives), dtype=np.int64)
        
        # Fill from right to left
        idx = self.max_seq_len - 1
        next_item = items[-1]
        
        for i in reversed(items[:-1]):
            seq[idx] = i
            pos[idx] = next_item
            
            for n in range(self.num_negatives):
                neg[idx, n] = self._sample_negative(user_item_set)
            
            next_item = i
            idx -= 1
            if idx < 0:
                break
        
        return {
            "user_id": torch.tensor(0, dtype=torch.long),  # Not used
            "input_seq": torch.tensor(seq, dtype=torch.long),
            "pos_items": torch.tensor(pos, dtype=torch.long),
            "neg_items": torch.tensor(neg.squeeze() if self.num_negatives == 1 else neg, dtype=torch.long),
        }
    
    def _get_eval_sample(
        self, 
        seq_items: List[int],
        target_items: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Get an evaluation sample."""
        target_item = target_items[0]
        
        # Build input sequence
        seq = np.zeros(self.max_seq_len, dtype=np.int64)
        idx = self.max_seq_len - 1
        
        for i in reversed(seq_items):
            seq[idx] = i
            idx -= 1
            if idx < 0:
                break
        
        return {
            "user_id": torch.tensor(0, dtype=torch.long),
            "input_seq": torch.tensor(seq, dtype=torch.long),
            "target_item": torch.tensor(target_item, dtype=torch.long),
        }


def get_lazy_dataloader(
    data_path: str,
    max_seq_len: int = 50,
    batch_size: int = 256,
    mode: str = "train",
    num_negatives: int = 1,
    num_workers: int = 4,
    shuffle: bool = None,
    cache_size: int = 1000,
) -> DataLoader:
    """
    Create a lazy-loading DataLoader for sequential recommendation.
    
    Args:
        data_path: Path to data file
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        mode: "train", "valid", or "test"
        num_negatives: Number of negative samples
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (default: True for train)
        cache_size: Number of sequences to cache
        
    Returns:
        DataLoader instance
    """
    dataset = LazySequentialDataset(
        data_path=data_path,
        max_seq_len=max_seq_len,
        mode=mode,
        num_negatives=num_negatives,
        cache_size=cache_size,
    )
    
    if shuffle is None:
        shuffle = (mode == "train")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(mode == "train"),
    )
