"""
Tests for loss functions.
"""

import pytest
import torch

from src.losses import BCELoss, BPRLoss, InfoNCELoss, DuoLoss


class TestBCELoss:
    """Tests for BCE loss."""
    
    def test_basic(self):
        loss_fn = BCELoss()
        
        pos_logits = torch.randn(4, 10)
        neg_logits = torch.randn(4, 10)
        
        loss = loss_fn(pos_logits, neg_logits)
        
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_with_mask(self):
        loss_fn = BCELoss()
        
        pos_logits = torch.randn(4, 10)
        neg_logits = torch.randn(4, 10)
        mask = torch.ones(4, 10)
        mask[:, :3] = 0  # Mask first 3 positions
        
        loss = loss_fn(pos_logits, neg_logits, mask)
        
        assert loss.dim() == 0
        assert loss >= 0


class TestBPRLoss:
    """Tests for BPR loss."""
    
    def test_basic(self):
        loss_fn = BPRLoss()
        
        pos_logits = torch.randn(4, 10)
        neg_logits = torch.randn(4, 10)
        
        loss = loss_fn(pos_logits, neg_logits)
        
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_positive_minus_negative(self):
        loss_fn = BPRLoss()
        
        # When pos >> neg, loss should be low
        pos_logits = torch.ones(4, 10) * 10
        neg_logits = torch.ones(4, 10) * -10
        
        loss_good = loss_fn(pos_logits, neg_logits)
        
        # When pos << neg, loss should be high
        pos_logits = torch.ones(4, 10) * -10
        neg_logits = torch.ones(4, 10) * 10
        
        loss_bad = loss_fn(pos_logits, neg_logits)
        
        assert loss_good < loss_bad


class TestInfoNCELoss:
    """Tests for InfoNCE loss."""
    
    def test_in_batch_negatives(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        
        query = torch.randn(8, 64)
        positive_key = torch.randn(8, 64)
        
        loss = loss_fn(query, positive_key)
        
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_explicit_negatives(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        
        query = torch.randn(8, 64)
        positive_key = torch.randn(8, 64)
        negative_keys = torch.randn(8, 10, 64)
        
        loss = loss_fn(query, positive_key, negative_keys)
        
        assert loss.dim() == 0
        assert loss >= 0


class TestDuoLoss:
    """Tests for DuoRec loss."""
    
    def test_unsupervised_contrastive(self):
        loss_fn = DuoLoss(temperature=1.0)
        
        repr_1 = torch.randn(8, 64)
        repr_2 = torch.randn(8, 64)
        
        unsup_loss, sup_loss = loss_fn(repr_1, repr_2)
        
        assert unsup_loss.dim() == 0
        assert unsup_loss >= 0
    
    def test_supervised_contrastive(self):
        loss_fn = DuoLoss(temperature=1.0)
        
        repr_1 = torch.randn(8, 64)
        repr_2 = torch.randn(8, 64)
        target_items = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])  # Some same targets
        
        unsup_loss, sup_loss = loss_fn(repr_1, repr_2, target_items)
        
        assert unsup_loss.dim() == 0
        # sup_loss may be 0 if no valid pairs
