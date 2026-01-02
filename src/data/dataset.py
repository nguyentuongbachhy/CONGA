"""
Dataset classes for sequential recommendation.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SequentialDataset(Dataset):
    """
    Dataset for sequential recommendation.
    
    Loads user-item interactions and creates training samples.
    """
    
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 50,
        mode: str = "train",
        num_negatives: int = 1,
    ):
        """
        Args:
            data_path: Path to data file (format: user_id item_id per line)
            max_seq_len: Maximum sequence length
            mode: "train", "valid", or "test"
            num_negatives: Number of negative samples per positive
        """
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.num_negatives = num_negatives

        # Evaluation: number of negative candidates per user (target + negatives)
        self.eval_num_negatives = 100
        
        # Load data
        self.user_seqs, self.num_users, self.num_items = self._load_data()
        
        # Split into train/valid/test
        self.train_seqs, self.valid_seqs, self.test_seqs = self._split_data()
        
        # Get valid users for this mode
        self.valid_users = self._get_valid_users()

        # Precompute evaluation negatives for determinism (valid/test only)
        self._eval_negatives = None
        if self.mode in {"valid", "test"}:
            self._eval_negatives = {}
            for user_id in self.valid_users:
                if self.mode == "valid":
                    seq_items = self.train_seqs[user_id]
                    target_item = self.valid_seqs[user_id][0]
                else:
                    seq_items = self.train_seqs[user_id] + self.valid_seqs[user_id]
                    target_item = self.test_seqs[user_id][0]

                interacted = set(seq_items)
                interacted.add(target_item)

                neg_items = np.zeros(self.eval_num_negatives, dtype=np.int64)
                for i in range(self.eval_num_negatives):
                    neg_items[i] = self._sample_negative(interacted)
                self._eval_negatives[user_id] = neg_items
    
    def _load_data(self) -> Tuple[Dict[int, List[int]], int, int]:
        """Load user-item interactions."""
        user_seqs = defaultdict(list)
        num_users = 0
        num_items = 0
        
        with open(self.data_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                user_id = int(parts[0])
                item_id = int(parts[1])
                
                user_seqs[user_id].append(item_id)
                num_users = max(num_users, user_id)
                num_items = max(num_items, item_id)
        
        return dict(user_seqs), num_users, num_items
    
    def _split_data(self) -> Tuple[Dict, Dict, Dict]:
        """Split sequences into train/valid/test."""
        train_seqs = {}
        valid_seqs = {}
        test_seqs = {}
        
        for user_id, items in self.user_seqs.items():
            if len(items) < 3:
                # Not enough items for split
                train_seqs[user_id] = items
                valid_seqs[user_id] = []
                test_seqs[user_id] = []
            else:
                train_seqs[user_id] = items[:-2]
                valid_seqs[user_id] = [items[-2]]
                test_seqs[user_id] = [items[-1]]
        
        return train_seqs, valid_seqs, test_seqs
    
    def _get_valid_users(self) -> List[int]:
        """Get users with sufficient interactions."""
        valid_users = []
        
        for user_id in self.user_seqs.keys():
            if self.mode == "train":
                if len(self.train_seqs[user_id]) >= 1:
                    valid_users.append(user_id)
            elif self.mode == "valid":
                if len(self.train_seqs[user_id]) >= 1 and len(self.valid_seqs[user_id]) >= 1:
                    valid_users.append(user_id)
            else:  # test
                if len(self.train_seqs[user_id]) >= 1 and len(self.test_seqs[user_id]) >= 1:
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
        
        if self.mode == "train":
            return self._get_train_sample(user_id)
        elif self.mode == "valid":
            return self._get_eval_sample(user_id, self.valid_seqs)
        else:
            return self._get_eval_sample(user_id, self.test_seqs)
    
    def _get_train_sample(self, user_id: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        items = self.train_seqs[user_id]
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
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "input_seq": torch.tensor(seq, dtype=torch.long),
            "pos_items": torch.tensor(pos, dtype=torch.long),
            "neg_items": torch.tensor(neg.squeeze() if self.num_negatives == 1 else neg, dtype=torch.long),
        }
    
    def _get_eval_sample(
        self, 
        user_id: int, 
        target_seqs: Dict[int, List[int]],
    ) -> Dict[str, torch.Tensor]:
        """Get an evaluation sample."""
        train_items = self.train_seqs[user_id]
        
        if self.mode == "valid":
            # Include all train items
            seq_items = train_items
        else:  # test
            # Include train + valid
            seq_items = train_items + self.valid_seqs[user_id]
        
        # Target item
        target_item = target_seqs[user_id][0]

        # Negative candidates (exclude all items user has interacted with + target)
        if self._eval_negatives is not None and user_id in self._eval_negatives:
            neg_items = self._eval_negatives[user_id]
        else:
            interacted = set(seq_items)
            interacted.add(target_item)
            neg_items = np.zeros(self.eval_num_negatives, dtype=np.int64)
            for i in range(self.eval_num_negatives):
                neg_items[i] = self._sample_negative(interacted)
        
        # Build input sequence
        seq = np.zeros(self.max_seq_len, dtype=np.int64)
        idx = self.max_seq_len - 1
        
        for i in reversed(seq_items):
            seq[idx] = i
            idx -= 1
            if idx < 0:
                break
        
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "input_seq": torch.tensor(seq, dtype=torch.long),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "neg_items": torch.tensor(neg_items, dtype=torch.long),
        }


def get_dataloader(
    data_path: str,
    max_seq_len: int = 50,
    batch_size: int = 256,
    mode: str = "train",
    num_negatives: int = 1,
    num_workers: int = 4,
    shuffle: bool = None,
) -> DataLoader:
    """
    Create a DataLoader for sequential recommendation.
    
    Args:
        data_path: Path to data file
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        mode: "train", "valid", or "test"
        num_negatives: Number of negative samples
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (default: True for train)
        
    Returns:
        DataLoader instance
    """
    dataset = SequentialDataset(
        data_path=data_path,
        max_seq_len=max_seq_len,
        mode=mode,
        num_negatives=num_negatives,
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


def load_dataset_info(data_path: str) -> Tuple[int, int, float]:
    """
    Load dataset statistics.
    
    Returns:
        num_users, num_items, avg_seq_len
    """
    dataset = SequentialDataset(data_path, mode="train")
    
    total_len = sum(len(seq) for seq in dataset.train_seqs.values())
    avg_len = total_len / len(dataset.train_seqs) if dataset.train_seqs else 0
    
    return dataset.num_users, dataset.num_items, avg_len
