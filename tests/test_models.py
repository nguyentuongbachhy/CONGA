"""
Tests for model implementations.
"""

import pytest
import torch

from src.models import get_model, SASRec, CL4SRec, GCL4SR, CONGA


class TestSASRec:
    """Tests for SASRec model."""
    
    @pytest.fixture
    def model(self):
        return SASRec(
            num_items=100,
            hidden_size=32,
            max_seq_len=20,
            num_layers=2,
            num_heads=2,
            dropout_rate=0.1,
            device="cpu",
        )
    
    def test_forward(self, model):
        batch_size = 4
        seq_len = 20
        
        item_seq = torch.randint(0, 100, (batch_size, seq_len))
        pos_items = torch.randint(1, 100, (batch_size, seq_len))
        neg_items = torch.randint(1, 100, (batch_size, seq_len))
        
        outputs = model(item_seq, pos_items, neg_items)
        
        assert "pos_logits" in outputs
        assert "neg_logits" in outputs
        assert outputs["pos_logits"].shape == (batch_size, seq_len)
        assert outputs["neg_logits"].shape == (batch_size, seq_len)
    
    def test_predict(self, model):
        batch_size = 4
        seq_len = 20
        
        item_seq = torch.randint(0, 100, (batch_size, seq_len))
        
        # Predict all items
        scores = model.predict(item_seq)
        assert scores.shape == (batch_size, 100)
        
        # Predict specific candidates
        candidates = torch.randint(1, 100, (batch_size, 10))
        scores = model.predict(item_seq, candidates)
        assert scores.shape == (batch_size, 10)
    
    def test_get_sequence_representation(self, model):
        batch_size = 4
        seq_len = 20
        
        item_seq = torch.randint(0, 100, (batch_size, seq_len))
        repr = model.get_sequence_representation(item_seq)
        
        assert repr.shape == (batch_size, 32)


class TestCL4SRec:
    """Tests for CL4SRec model."""
    
    @pytest.fixture
    def model(self):
        return CL4SRec(
            num_items=100,
            hidden_size=32,
            max_seq_len=20,
            num_layers=2,
            num_heads=2,
            dropout_rate=0.1,
            device="cpu",
            contrastive_weight=0.1,
        )
    
    def test_forward_with_contrastive(self, model):
        batch_size = 4
        seq_len = 20
        
        item_seq = torch.randint(0, 100, (batch_size, seq_len))
        pos_items = torch.randint(1, 100, (batch_size, seq_len))
        neg_items = torch.randint(1, 100, (batch_size, seq_len))
        
        outputs = model(item_seq, pos_items, neg_items)
        
        assert "pos_logits" in outputs
        assert "neg_logits" in outputs
        assert "cl_loss" in outputs
        assert outputs["cl_loss"].dim() == 0  # Scalar


class TestCONGA:
    """Tests for CONGA model."""
    
    @pytest.fixture
    def model(self):
        return CONGA(
            num_items=100,
            hidden_size=32,
            max_seq_len=20,
            num_layers=2,
            num_heads=2,
            dropout_rate=0.1,
            device="cpu",
            num_local_layers=2,
            num_global_layers=1,
            memory_bank_size=100,
            contrastive_weight=0.1,
            graph_cl_weight=0.1,
        )
    
    def test_forward(self, model):
        batch_size = 4
        seq_len = 20
        
        item_seq = torch.randint(0, 100, (batch_size, seq_len))
        pos_items = torch.randint(1, 100, (batch_size, seq_len))
        neg_items = torch.randint(1, 100, (batch_size, seq_len))
        
        outputs = model(item_seq, pos_items, neg_items)
        
        assert "pos_logits" in outputs
        assert "neg_logits" in outputs
        assert "seq_cl_loss" in outputs
        assert "graph_cl_loss" in outputs
    
    def test_nested_graph_components(self, model):
        batch_size = 4
        seq_len = 20
        
        item_seq = torch.randint(0, 100, (batch_size, seq_len))
        pos_items = torch.randint(1, 100, (batch_size, seq_len))
        neg_items = torch.randint(1, 100, (batch_size, seq_len))
        
        outputs = model(item_seq, pos_items, neg_items)
        
        assert "local_repr" in outputs
        assert "global_repr" in outputs
        assert outputs["local_repr"].shape == (batch_size, 32)
        assert outputs["global_repr"].shape == (batch_size, 32)


class TestModelFactory:
    """Tests for model factory function."""
    
    @pytest.mark.parametrize("model_name", ["sasrec", "cl4srec", "gcl4sr", "conga"])
    def test_get_model(self, model_name):
        model = get_model(
            model_name=model_name,
            num_items=100,
            hidden_size=32,
            max_seq_len=20,
            num_layers=2,
            num_heads=2,
            dropout_rate=0.1,
            device="cpu",
        )
        
        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "predict")
    
    def test_unknown_model(self):
        with pytest.raises(ValueError):
            get_model(model_name="unknown", num_items=100)
