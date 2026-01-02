"""
Data augmentation strategies for sequential recommendation.
Based on CL4SRec paper.
"""

import random
from typing import List, Optional
import torch
import numpy as np


class SequenceAugmentor:
    """
    Sequence augmentation for contrastive learning.
    
    Implements three augmentation strategies:
    1. Item Crop: randomly crop a contiguous subsequence
    2. Item Mask: randomly mask items
    3. Item Reorder: randomly reorder a contiguous subsequence
    """
    
    def __init__(
        self,
        crop_ratio: float = 0.6,
        mask_ratio: float = 0.3,
        reorder_ratio: float = 0.6,
        mask_token: int = 0,
    ):
        """
        Args:
            crop_ratio: Ratio of items to keep when cropping
            mask_ratio: Ratio of items to mask
            reorder_ratio: Ratio of items to reorder
            mask_token: Token to use for masking (default: padding token)
        """
        self.crop_ratio = crop_ratio
        self.mask_ratio = mask_ratio
        self.reorder_ratio = reorder_ratio
        self.mask_token = mask_token
    
    def crop(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Randomly crop a contiguous subsequence.
        
        Args:
            seq: [batch_size, seq_len] or [seq_len]
            
        Returns:
            Cropped sequence (right-aligned with padding)
        """
        if seq.dim() == 1:
            return self._crop_single(seq)
        
        batch_size, seq_len = seq.shape
        result = torch.zeros_like(seq)
        
        for i in range(batch_size):
            result[i] = self._crop_single(seq[i])
        
        return result
    
    def _crop_single(self, seq: torch.Tensor) -> torch.Tensor:
        """Crop a single sequence."""
        seq_len = seq.shape[0]
        valid_mask = seq != 0
        valid_items = seq[valid_mask]
        valid_len = len(valid_items)
        
        if valid_len <= 1:
            return seq.clone()
        
        # Determine crop length
        crop_len = max(1, int(valid_len * self.crop_ratio))
        
        # Random start position
        start = random.randint(0, valid_len - crop_len)
        cropped_items = valid_items[start:start + crop_len]
        
        # Right-align
        result = torch.zeros(seq_len, dtype=seq.dtype, device=seq.device)
        result[-crop_len:] = cropped_items
        
        return result
    
    def mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Randomly mask items in the sequence.
        
        Args:
            seq: [batch_size, seq_len] or [seq_len]
            
        Returns:
            Masked sequence
        """
        if seq.dim() == 1:
            return self._mask_single(seq)
        
        batch_size, seq_len = seq.shape
        result = seq.clone()
        
        for i in range(batch_size):
            result[i] = self._mask_single(seq[i])
        
        return result
    
    def _mask_single(self, seq: torch.Tensor) -> torch.Tensor:
        """Mask a single sequence."""
        result = seq.clone()
        valid_indices = (seq != 0).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) <= 1:
            return result
        
        # Determine number of items to mask
        num_mask = max(1, int(len(valid_indices) * self.mask_ratio))
        
        # Random mask indices (keep at least one item)
        mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_mask]]
        result[mask_indices] = self.mask_token
        
        return result
    
    def reorder(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Randomly reorder a contiguous subsequence.
        
        Args:
            seq: [batch_size, seq_len] or [seq_len]
            
        Returns:
            Reordered sequence
        """
        if seq.dim() == 1:
            return self._reorder_single(seq)
        
        batch_size, seq_len = seq.shape
        result = seq.clone()
        
        for i in range(batch_size):
            result[i] = self._reorder_single(seq[i])
        
        return result
    
    def _reorder_single(self, seq: torch.Tensor) -> torch.Tensor:
        """Reorder a single sequence."""
        result = seq.clone()
        valid_indices = (seq != 0).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) <= 2:
            return result
        
        valid_len = len(valid_indices)
        
        # Determine reorder length
        reorder_len = max(2, int(valid_len * self.reorder_ratio))
        
        # Random start position
        start = random.randint(0, valid_len - reorder_len)
        reorder_indices = valid_indices[start:start + reorder_len]
        
        # Shuffle items at these positions
        items = result[reorder_indices].clone()
        perm = torch.randperm(len(items))
        result[reorder_indices] = items[perm]
        
        return result
    
    def substitute(
        self, 
        seq: torch.Tensor, 
        num_items: int,
        sub_ratio: float = 0.2,
    ) -> torch.Tensor:
        """
        Randomly substitute items with random items.
        
        Args:
            seq: [batch_size, seq_len] or [seq_len]
            num_items: Total number of items in dataset
            sub_ratio: Ratio of items to substitute
            
        Returns:
            Substituted sequence
        """
        if seq.dim() == 1:
            return self._substitute_single(seq, num_items, sub_ratio)
        
        batch_size, seq_len = seq.shape
        result = seq.clone()
        
        for i in range(batch_size):
            result[i] = self._substitute_single(seq[i], num_items, sub_ratio)
        
        return result
    
    def _substitute_single(
        self, 
        seq: torch.Tensor, 
        num_items: int,
        sub_ratio: float,
    ) -> torch.Tensor:
        """Substitute a single sequence."""
        result = seq.clone()
        valid_indices = (seq != 0).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) <= 1:
            return result
        
        # Determine number of items to substitute
        num_sub = max(1, int(len(valid_indices) * sub_ratio))
        sub_indices = valid_indices[torch.randperm(len(valid_indices))[:num_sub]]
        
        # Random items
        random_items = torch.randint(1, num_items + 1, (num_sub,), device=seq.device)
        result[sub_indices] = random_items
        
        return result
    
    def insert(
        self, 
        seq: torch.Tensor, 
        num_items: int,
        insert_ratio: float = 0.2,
    ) -> torch.Tensor:
        """
        Randomly insert items into the sequence.
        
        Args:
            seq: [batch_size, seq_len] or [seq_len]
            num_items: Total number of items in dataset
            insert_ratio: Ratio of items to insert
            
        Returns:
            Sequence with insertions (may truncate)
        """
        if seq.dim() == 1:
            return self._insert_single(seq, num_items, insert_ratio)
        
        batch_size, seq_len = seq.shape
        result = seq.clone()
        
        for i in range(batch_size):
            result[i] = self._insert_single(seq[i], num_items, insert_ratio)
        
        return result
    
    def _insert_single(
        self, 
        seq: torch.Tensor, 
        num_items: int,
        insert_ratio: float,
    ) -> torch.Tensor:
        """Insert items into a single sequence."""
        seq_len = seq.shape[0]
        valid_mask = seq != 0
        valid_items = seq[valid_mask].tolist()
        
        if len(valid_items) <= 1:
            return seq.clone()
        
        # Determine number of items to insert
        num_insert = max(1, int(len(valid_items) * insert_ratio))
        
        # Insert random items at random positions
        for _ in range(num_insert):
            pos = random.randint(0, len(valid_items))
            item = random.randint(1, num_items)
            valid_items.insert(pos, item)
        
        # Truncate to max_seq_len and right-align
        if len(valid_items) > seq_len:
            valid_items = valid_items[-seq_len:]
        
        result = torch.zeros(seq_len, dtype=seq.dtype, device=seq.device)
        result[-len(valid_items):] = torch.tensor(valid_items, dtype=seq.dtype, device=seq.device)
        
        return result
    
    def augment(
        self, 
        seq: torch.Tensor, 
        aug_types: Optional[List[str]] = None,
        num_items: int = None,
    ) -> torch.Tensor:
        """
        Apply a random augmentation.
        
        Args:
            seq: Input sequence
            aug_types: List of augmentation types to choose from
            num_items: Total number of items (needed for substitute/insert)
            
        Returns:
            Augmented sequence
        """
        if aug_types is None:
            aug_types = ["crop", "mask", "reorder"]
        
        aug_type = random.choice(aug_types)
        
        if aug_type == "crop":
            return self.crop(seq)
        elif aug_type == "mask":
            return self.mask(seq)
        elif aug_type == "reorder":
            return self.reorder(seq)
        elif aug_type == "substitute" and num_items is not None:
            return self.substitute(seq, num_items)
        elif aug_type == "insert" and num_items is not None:
            return self.insert(seq, num_items)
        else:
            return seq.clone()
    
    def get_two_views(
        self, 
        seq: torch.Tensor,
        aug_types: Optional[List[str]] = None,
        num_items: int = None,
    ) -> tuple:
        """
        Generate two augmented views of the sequence.
        
        Args:
            seq: Input sequence
            aug_types: Augmentation types
            num_items: Total items for substitute/insert
            
        Returns:
            (view1, view2) tuple
        """
        view1 = self.augment(seq, aug_types, num_items)
        view2 = self.augment(seq, aug_types, num_items)
        return view1, view2
