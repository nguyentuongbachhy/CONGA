"""
Pattern-based utilities for improving sequential recommendation training

Three main approaches:
1. Pattern-aware Initialization - Initialize embeddings from co-occurrence patterns
2. Pattern Regularization - Add pattern consistency loss
3. Pattern-guided Negative Sampling - Sample negatives based on patterns
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict


class PatternAwareInitializer:
    """
    Initialize item embeddings using co-occurrence patterns
    
    Items that frequently co-occur should have similar embeddings.
    This provides better initialization than random.
    """
    
    def __init__(self, pattern_file: str):
        """
        Args:
            pattern_file: Path to patterns.pkl file
        """
        with open(pattern_file, 'rb') as f:
            data = pickle.load(f)
        
        self.patterns = data['patterns']
        self.stats = data['stats']
        
        # Build co-occurrence matrix
        self.cooccurrence = defaultdict(lambda: defaultdict(float))
        for pattern, support in self.patterns:
            if len(pattern) == 2:
                item1, item2 = pattern
                self.cooccurrence[item1][item2] = support
                self.cooccurrence[item2][item1] = support
        
        print(f"  Loaded {len(self.patterns)} patterns")
        print(f"  Co-occurrence pairs: {sum(len(v) for v in self.cooccurrence.values())}")
    
    def initialize_embeddings(self, item_emb: nn.Embedding, alpha: float = 0.3):
        """
        Initialize embeddings to reflect co-occurrence patterns
        
        Strategy: For each item, average embeddings of co-occurring items
        weighted by co-occurrence strength
        
        Args:
            item_emb: Item embedding layer to initialize
            alpha: Weight for pattern-based initialization (0=random, 1=full pattern)
        """
        print(f"\n  Pattern-aware Initialization (alpha={alpha})")
        
        with torch.no_grad():
            original_emb = item_emb.weight.data.clone()
            num_items = item_emb.num_embeddings
            
            # For each item, compute pattern-based embedding
            for item in range(1, num_items):  # Skip padding (0)
                if item not in self.cooccurrence:
                    continue
                
                # Get co-occurring items and their weights
                neighbors = self.cooccurrence[item]
                if not neighbors:
                    continue
                
                # Weighted average of neighbor embeddings
                total_weight = sum(neighbors.values())
                pattern_emb = torch.zeros_like(item_emb.weight[item])
                
                for neighbor, weight in neighbors.items():
                    if neighbor < num_items:
                        pattern_emb += (weight / total_weight) * original_emb[neighbor]
                
                # Blend with original (random) embedding
                item_emb.weight[item] = (1 - alpha) * original_emb[item] + alpha * pattern_emb
            
            # Keep padding as zero
            item_emb.weight[0] = 0
        
        print(f"  ✓ Initialized {len(self.cooccurrence)} items with pattern information")


class PatternRegularizer:
    """
    Add regularization loss to encourage pattern consistency
    
    Items in frequent patterns should have similar embeddings.
    """
    
    def __init__(self, pattern_file: str, top_k: int = 500):
        """
        Args:
            pattern_file: Path to patterns.pkl file
            top_k: Use only top-k patterns for efficiency
        """
        with open(pattern_file, 'rb') as f:
            data = pickle.load(f)
        
        # Use only top-k patterns
        self.patterns = data['patterns'][:top_k]
        
        # Build pattern pairs with weights
        self.pattern_pairs = []
        for pattern, support in self.patterns:
            if len(pattern) >= 2:
                # Create all pairs within pattern
                for i in range(len(pattern)):
                    for j in range(i + 1, len(pattern)):
                        self.pattern_pairs.append((pattern[i], pattern[j], support))
        
        print(f"  Loaded {len(self.pattern_pairs)} pattern pairs for regularization")
    
    def compute_loss(self, item_emb: nn.Embedding, device: torch.device) -> torch.Tensor:
        """
        Compute pattern consistency loss
        
        Loss = sum over patterns: weight * ||emb(item1) - emb(item2)||^2
        
        Args:
            item_emb: Item embedding layer
            device: Device to compute on
            
        Returns:
            Pattern regularization loss
        """
        if not self.pattern_pairs:
            return torch.tensor(0.0, device=device)
        
        # Sample a subset for efficiency (avoid computing all pairs every batch)
        sample_size = min(100, len(self.pattern_pairs))
        sampled_pairs = np.random.choice(len(self.pattern_pairs), sample_size, replace=False)
        
        total_loss = 0.0
        total_weight = 0.0
        
        for idx in sampled_pairs:
            item1, item2, weight = self.pattern_pairs[idx]
            
            # Get embeddings
            emb1 = item_emb.weight[item1]
            emb2 = item_emb.weight[item2]
            
            # L2 distance
            dist = torch.sum((emb1 - emb2) ** 2)
            
            # Weighted loss (higher support = should be more similar)
            total_loss += weight * dist
            total_weight += weight
        
        if total_weight > 0:
            return total_loss / total_weight
        else:
            return torch.tensor(0.0, device=device)


class PatternGuidedNegativeSampler:
    """
    Sample negative items guided by patterns
    
    Strategy: Mix random negatives with pattern-based negatives
    - Random negatives: Standard approach
    - Pattern negatives: Items that co-occur with positive but not in same pattern
                        (hard negatives that are semantically related)
    """
    
    def __init__(self, pattern_file: str, num_items: int):
        """
        Args:
            pattern_file: Path to patterns.pkl file
            num_items: Total number of items
        """
        with open(pattern_file, 'rb') as f:
            data = pickle.load(f)
        
        self.patterns = data['patterns']
        self.num_items = num_items
        
        # Build item -> related items mapping
        self.item_neighbors = defaultdict(set)
        for pattern, support in self.patterns:
            for item in pattern:
                self.item_neighbors[item].update(pattern)
                self.item_neighbors[item].discard(item)  # Remove self
        
        print(f"  Built neighbor index for {len(self.item_neighbors)} items")
    
    def sample_negatives(self, pos_items: List[int], num_negatives: int, 
                        pattern_ratio: float = 0.3) -> List[int]:
        """
        Sample negative items with pattern guidance
        
        Args:
            pos_items: List of positive items in sequence
            num_negatives: Number of negatives to sample
            pattern_ratio: Ratio of pattern-based negatives (0.3 = 30% pattern, 70% random)
            
        Returns:
            List of negative item IDs
        """
        num_pattern_negs = int(num_negatives * pattern_ratio)
        num_random_negs = num_negatives - num_pattern_negs
        
        negatives = []
        pos_set = set(pos_items)
        
        # Sample pattern-based negatives (hard negatives)
        if num_pattern_negs > 0:
            # Get neighbors of positive items
            candidate_neighbors = set()
            for pos_item in pos_items:
                if pos_item in self.item_neighbors:
                    candidate_neighbors.update(self.item_neighbors[pos_item])
            
            # Remove positive items
            candidate_neighbors -= pos_set
            
            if candidate_neighbors:
                # Sample from neighbors
                sampled = np.random.choice(
                    list(candidate_neighbors),
                    min(num_pattern_negs, len(candidate_neighbors)),
                    replace=False
                )
                negatives.extend(sampled)
        
        # Fill remaining with random negatives
        remaining = num_negatives - len(negatives)
        if remaining > 0:
            while len(negatives) < num_negatives:
                neg = np.random.randint(1, self.num_items + 1)
                if neg not in pos_set and neg not in negatives:
                    negatives.append(neg)
        
        return negatives[:num_negatives]


def load_patterns(pattern_file: str):
    """Load patterns from file"""
    with open(pattern_file, 'rb') as f:
        return pickle.load(f)


def print_pattern_stats(pattern_file: str):
    """Print statistics about patterns"""
    data = load_patterns(pattern_file)
    
    print("\n" + "="*60)
    print("PATTERN STATISTICS")
    print("="*60)
    print(f"Total patterns: {len(data['patterns'])}")
    
    if 'stats' in data:
        stats = data['stats']
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {stats.get('num_nodes', 'N/A')}")
        print(f"  Edges: {stats.get('num_edges', 'N/A')}")
        print(f"  Avg degree: {stats.get('avg_degree', 'N/A'):.2f}")
        
        if 'pattern_length_distribution' in stats:
            print(f"\nPattern Length Distribution:")
            for length, count in sorted(stats['pattern_length_distribution'].items()):
                print(f"  Length {length}: {count} patterns")
    
    print(f"\nTop 10 Patterns:")
    for i, (pattern, support) in enumerate(data['patterns'][:10], 1):
        print(f"  {i}. {pattern} (support: {support})")
    
    print("="*60 + "\n")
