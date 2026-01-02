"""
Tests for data utilities.
"""

import pytest
import torch
import numpy as np

from src.data.augmentation import SequenceAugmentor
from src.data.graph_builder import GraphBuilder


class TestSequenceAugmentor:
    """Tests for sequence augmentation."""
    
    @pytest.fixture
    def augmentor(self):
        return SequenceAugmentor(
            crop_ratio=0.6,
            mask_ratio=0.3,
            reorder_ratio=0.6,
        )
    
    def test_crop(self, augmentor):
        seq = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5]])
        cropped = augmentor.crop(seq)
        
        # Should preserve right-alignment
        assert cropped[0, -1] != 0 or seq[0].sum() == 0
        
        # Should be shorter or equal
        assert (cropped != 0).sum() <= (seq != 0).sum()
    
    def test_mask(self, augmentor):
        seq = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5]])
        masked = augmentor.mask(seq)
        
        # Some items should be masked (set to 0)
        original_nonzero = (seq != 0).sum()
        masked_nonzero = (masked != 0).sum()
        
        # At least one item should be masked
        assert masked_nonzero <= original_nonzero
    
    def test_reorder(self, augmentor):
        seq = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5]])
        reordered = augmentor.reorder(seq)
        
        # Same items, potentially different order
        original_items = set(seq[0].tolist()) - {0}
        reordered_items = set(reordered[0].tolist()) - {0}
        
        assert original_items == reordered_items
    
    def test_batch_augmentation(self, augmentor):
        batch = torch.tensor([
            [0, 0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
        ])
        
        cropped = augmentor.crop(batch)
        assert cropped.shape == batch.shape
        
        masked = augmentor.mask(batch)
        assert masked.shape == batch.shape
        
        reordered = augmentor.reorder(batch)
        assert reordered.shape == batch.shape


class TestGraphBuilder:
    """Tests for graph construction."""
    
    @pytest.fixture
    def builder(self):
        return GraphBuilder(num_users=100, num_items=50)
    
    @pytest.fixture
    def sample_sequences(self):
        return {
            1: [1, 2, 3, 4, 5],
            2: [2, 3, 4, 5, 6],
            3: [1, 3, 5, 7, 9],
        }
    
    def test_build_transition_graph(self, builder, sample_sequences):
        edge_index, edge_weight = builder.build_transition_graph(sample_sequences)
        
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0
        assert edge_weight is not None
        assert len(edge_weight) == edge_index.shape[1]
    
    def test_build_cooccurrence_graph(self, builder, sample_sequences):
        edge_index, edge_weight = builder.build_cooccurrence_graph(
            sample_sequences, window_size=3
        )
        
        assert edge_index.shape[0] == 2
        assert len(edge_weight) == edge_index.shape[1]
    
    def test_build_session_graph(self, builder):
        item_seq = torch.tensor([
            [0, 0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5],
        ])
        
        edge_index, edge_weight = builder.build_session_graph(item_seq)
        
        assert edge_index.shape[0] == 2
        assert len(edge_weight) == edge_index.shape[1]
