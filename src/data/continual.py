"""
Continual learning data stream utilities.
"""

import random
from typing import Dict, List, Tuple, Iterator, Optional
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ContinualDataStream:
    """
    Simulates a continual data stream for sequential recommendation.
    
    Splits data into time-based chunks and provides methods
    for incremental training.
    """
    
    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        num_chunks: int = 5,
        chunk_strategy: str = "time",
    ):
        """
        Args:
            user_sequences: Dict of user_id -> item list
            num_chunks: Number of temporal chunks
            chunk_strategy: "time" (temporal split) or "random"
        """
        self.user_sequences = user_sequences
        self.num_chunks = num_chunks
        self.chunk_strategy = chunk_strategy
        
        self.chunks = self._create_chunks()
        self.current_chunk = 0
    
    def _create_chunks(self) -> List[Dict[int, List[int]]]:
        """Split data into chunks."""
        chunks = [defaultdict(list) for _ in range(self.num_chunks)]
        
        for user_id, items in self.user_sequences.items():
            if len(items) < self.num_chunks:
                # Put all in first chunk
                chunks[0][user_id] = items
            else:
                # Split by position (simulating time)
                chunk_size = len(items) // self.num_chunks
                
                for i in range(self.num_chunks):
                    start = i * chunk_size
                    if i == self.num_chunks - 1:
                        end = len(items)
                    else:
                        end = (i + 1) * chunk_size
                    
                    chunks[i][user_id] = items[start:end]
        
        return [dict(c) for c in chunks]
    
    def get_chunk(self, chunk_idx: int) -> Dict[int, List[int]]:
        """Get a specific chunk."""
        if 0 <= chunk_idx < len(self.chunks):
            return self.chunks[chunk_idx]
        raise IndexError(f"Chunk index {chunk_idx} out of range")
    
    def get_cumulative_data(self, up_to_chunk: int) -> Dict[int, List[int]]:
        """Get all data up to and including a chunk."""
        cumulative = defaultdict(list)
        
        for i in range(up_to_chunk + 1):
            for user_id, items in self.chunks[i].items():
                cumulative[user_id].extend(items)
        
        return dict(cumulative)
    
    def get_next_chunk(self) -> Optional[Dict[int, List[int]]]:
        """Get the next chunk in sequence."""
        if self.current_chunk >= len(self.chunks):
            return None
        
        chunk = self.chunks[self.current_chunk]
        self.current_chunk += 1
        return chunk
    
    def reset(self) -> None:
        """Reset to first chunk."""
        self.current_chunk = 0
    
    def __iter__(self) -> Iterator[Dict[int, List[int]]]:
        """Iterate over chunks."""
        for chunk in self.chunks:
            yield chunk
    
    def __len__(self) -> int:
        return len(self.chunks)


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.
    
    Stores representative samples from previous tasks.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        max_seq_len: int = 50,
        strategy: str = "reservoir",
    ):
        """
        Args:
            buffer_size: Maximum number of samples to store
            max_seq_len: Maximum sequence length
            strategy: "reservoir" (random), "priority", or "herding"
        """
        self.buffer_size = buffer_size
        self.max_seq_len = max_seq_len
        self.strategy = strategy
        
        # Storage
        self.sequences = []
        self.targets = []
        self.priorities = []
        
        # For reservoir sampling
        self.seen_count = 0
    
    def add(
        self,
        sequence: np.ndarray,
        target: int,
        priority: float = 1.0,
    ) -> None:
        """Add a sample to the buffer."""
        if self.strategy == "reservoir":
            self._reservoir_add(sequence, target)
        elif self.strategy == "priority":
            self._priority_add(sequence, target, priority)
        else:
            self._simple_add(sequence, target)
    
    def _simple_add(self, sequence: np.ndarray, target: int) -> None:
        """Simple FIFO addition."""
        if len(self.sequences) >= self.buffer_size:
            self.sequences.pop(0)
            self.targets.pop(0)
        
        self.sequences.append(sequence.copy())
        self.targets.append(target)
    
    def _reservoir_add(self, sequence: np.ndarray, target: int) -> None:
        """Reservoir sampling for uniform random sampling."""
        self.seen_count += 1
        
        if len(self.sequences) < self.buffer_size:
            self.sequences.append(sequence.copy())
            self.targets.append(target)
        else:
            # Random replacement
            idx = random.randint(0, self.seen_count - 1)
            if idx < self.buffer_size:
                self.sequences[idx] = sequence.copy()
                self.targets[idx] = target
    
    def _priority_add(
        self, 
        sequence: np.ndarray, 
        target: int,
        priority: float,
    ) -> None:
        """Priority-based addition."""
        if len(self.sequences) < self.buffer_size:
            self.sequences.append(sequence.copy())
            self.targets.append(target)
            self.priorities.append(priority)
        else:
            # Replace lowest priority
            min_idx = np.argmin(self.priorities)
            if priority > self.priorities[min_idx]:
                self.sequences[min_idx] = sequence.copy()
                self.targets[min_idx] = target
                self.priorities[min_idx] = priority
    
    def sample(
        self, 
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch from the buffer."""
        if len(self.sequences) == 0:
            return None, None
        
        batch_size = min(batch_size, len(self.sequences))
        
        if self.strategy == "priority" and self.priorities:
            # Weighted sampling by priority
            probs = np.array(self.priorities)
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.sequences), batch_size, p=probs)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.sequences), batch_size, replace=False)
        
        sequences = np.stack([self.sequences[i] for i in indices])
        targets = np.array([self.targets[i] for i in indices])
        
        return sequences, targets
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.sequences = []
        self.targets = []
        self.priorities = []
        self.seen_count = 0


class ContinualDataset(Dataset):
    """
    Dataset wrapper for continual learning.
    """
    
    def __init__(
        self,
        current_data: Dict[int, List[int]],
        replay_buffer: Optional[ReplayBuffer] = None,
        replay_ratio: float = 0.3,
        max_seq_len: int = 50,
        num_items: int = None,
    ):
        """
        Args:
            current_data: Current chunk data
            replay_buffer: Buffer of past samples
            replay_ratio: Ratio of replay samples in each batch
            max_seq_len: Maximum sequence length
            num_items: Total number of items
        """
        self.current_data = current_data
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        
        # Prepare current samples
        self.current_samples = self._prepare_samples(current_data)
    
    def _prepare_samples(
        self, 
        data: Dict[int, List[int]]
    ) -> List[Tuple[np.ndarray, int]]:
        """Prepare training samples."""
        samples = []
        
        for user_id, items in data.items():
            if len(items) < 2:
                continue
            
            # Create sequence and target
            seq = np.zeros(self.max_seq_len, dtype=np.int64)
            
            # Use all items except last as input
            input_items = items[:-1]
            seq_start = max(0, self.max_seq_len - len(input_items))
            seq[seq_start:] = input_items[-(self.max_seq_len):]
            
            target = items[-1]
            samples.append((seq, target))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.current_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Decide if using replay
        use_replay = (
            self.replay_buffer is not None and 
            len(self.replay_buffer) > 0 and
            random.random() < self.replay_ratio
        )
        
        if use_replay:
            # Sample from replay buffer
            seqs, targets = self.replay_buffer.sample(1)
            seq = seqs[0]
            target = targets[0]
        else:
            # Use current sample
            seq, target = self.current_samples[idx]
        
        # Sample negative
        neg = random.randint(1, self.num_items or 10000)
        while neg == target:
            neg = random.randint(1, self.num_items or 10000)
        
        return {
            "input_seq": torch.tensor(seq, dtype=torch.long),
            "target_item": torch.tensor(target, dtype=torch.long),
            "neg_item": torch.tensor(neg, dtype=torch.long),
        }
