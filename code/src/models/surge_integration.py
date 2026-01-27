import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any


class SURGEGraphInjector(nn.Module):
    """
    SURGE-style: Inject graph embeddings directly into SASRec self-attention
    Nested integration approach for continual sequential recommendation
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 1,
        injection_mode: str = "concat",  # concat, gate, residual
        graph_weight: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.injection_mode = injection_mode
        self.graph_weight = graph_weight
        
        if injection_mode == "gate":
            # Gating mechanism to control graph influence
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )
        elif injection_mode == "concat":
            # Projection after concatenation
            self.proj = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(
        self,
        seq_emb: torch.Tensor,
        graph_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject graph embeddings into sequence embeddings
        
        Args:
            seq_emb: [B, L, D] sequence embeddings from item_emb
            graph_emb: [B, L, D] graph embeddings for the same items
            
        Returns:
            fused_emb: [B, L, D] fused embeddings
        """
        if self.injection_mode == "residual":
            # Simple weighted residual
            return seq_emb + self.graph_weight * graph_emb
        
        elif self.injection_mode == "gate":
            # Gated fusion
            concat = torch.cat([seq_emb, graph_emb], dim=-1)
            gate_value = self.gate(concat)
            return gate_value * seq_emb + (1 - gate_value) * graph_emb
        
        elif self.injection_mode == "concat":
            # Concatenate and project
            concat = torch.cat([seq_emb, graph_emb], dim=-1)
            return self.proj(concat)
        
        else:
            return seq_emb


class GraphEnhancedSASRec(nn.Module):
    """
    SASRec with SURGE-style graph injection
    Integrates KGLN graph embeddings directly into self-attention
    """
    
    def __init__(
        self,
        sasrec_model: nn.Module,
        graph_embeddings: torch.Tensor,
        injection_mode: str = "gate",
        graph_weight: float = 0.3,
        freeze_graph: bool = True,
    ):
        super().__init__()
        self.sasrec = sasrec_model
        self.injection_mode = injection_mode
        self.graph_weight = graph_weight
        
        # Get embedding dimension from SASRec
        item_emb = getattr(sasrec_model, 'item_emb')
        self.embedding_dim = int(item_emb.embedding_dim)
        
        # Store graph embeddings
        if freeze_graph:
            self.register_buffer('graph_embeddings', graph_embeddings)
        else:
            self.graph_embeddings = nn.Parameter(graph_embeddings)
        
        # SURGE injector
        num_heads = getattr(sasrec_model, 'num_heads', 1) if hasattr(sasrec_model, 'num_heads') else 1
        self.injector = SURGEGraphInjector(
            self.embedding_dim,
            num_heads=num_heads,
            injection_mode=injection_mode,
            graph_weight=graph_weight,
        )
        
        print(f"[SURGE Integration] Mode={injection_mode}, Weight={graph_weight}, Freeze={freeze_graph}")
    
    def get_fused_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get fused sequence + graph embeddings"""
        item_emb = getattr(self.sasrec, 'item_emb')
        seq_emb = item_emb(item_ids)
        graph_emb = self.graph_embeddings[item_ids].to(seq_emb.device)
        
        return self.injector(seq_emb, graph_emb)
    
    def log2feats(self, log_seqs: Any) -> torch.Tensor:
        """Override log2feats to inject graph embeddings"""
        item_ids = torch.as_tensor(log_seqs, dtype=torch.long, device=self.sasrec.dev)
        
        # Get fused embeddings
        seqs = self.get_fused_embeddings(item_ids)
        
        # Scale and dropout (same as original SASRec)
        seqs *= self.embedding_dim ** 0.5
        seqs = self.sasrec.emb_dropout(seqs)
        
        # Multi-stream processing
        x_streams = seqs.unsqueeze(2).repeat(1, 1, self.sasrec.num_streams, 1)
        
        for i in range(len(self.sasrec.attention_layers)):
            layer_idx = i
            
            def make_attn_wrapper(idx: int):
                def attn_wrapper(x: torch.Tensor) -> torch.Tensor:
                    if self.sasrec.norm_first:
                        x = self.sasrec.attention_layernorms[idx](x)
                    return self.sasrec.attention_layers[idx](
                        x, attn_mask=None, rotary_emb_fn=self.sasrec.rope
                    )
                return attn_wrapper
            
            x_streams = self.sasrec.mhc_attn_layers[layer_idx](x_streams, make_attn_wrapper(layer_idx))
            
            # FFN wrapper
            def make_ffn_wrapper(idx: int, is_stem: bool):
                def ffn_wrapper(x: torch.Tensor) -> torch.Tensor:
                    if self.sasrec.norm_first:
                        x = self.sasrec.forward_layernorms[idx](x)
                    if is_stem:
                        return self.sasrec.forward_layers[idx](x, token_ids=item_ids)
                    else:
                        return self.sasrec.forward_layers[idx](x)
                return ffn_wrapper
            
            is_stem_layer = layer_idx in self.sasrec.stem_layers
            x_streams = self.sasrec.mhc_ffn_layers[layer_idx](x_streams, make_ffn_wrapper(layer_idx, is_stem_layer))
        
        final_seqs = torch.mean(x_streams, dim=2)
        log_feats = self.sasrec.last_layernorm(final_seqs)
        
        return log_feats
    
    def forward(
        self,
        user_ids: Any,
        log_seqs: Any,
        pos_seqs: Any,
        neg_seqs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with graph-enhanced embeddings"""
        log_feats = self.log2feats(log_seqs)
        
        # Use original item embeddings for targets (not graph-enhanced)
        item_emb = getattr(self.sasrec, 'item_emb')
        pos_embs = item_emb(torch.as_tensor(pos_seqs, dtype=torch.long, device=self.sasrec.dev))
        neg_ids = torch.as_tensor(neg_seqs, dtype=torch.long, device=self.sasrec.dev)
        neg_embs = item_emb(neg_ids)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        
        if neg_embs.dim() == 3:
            neg_logits = (log_feats * neg_embs).sum(dim=-1)
        else:
            neg_logits = (log_feats.unsqueeze(-2) * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits
    
    def predict(self, user_ids: Any, log_seqs: Any, item_indices: Any) -> torch.Tensor:
        """Prediction with graph-enhanced features"""
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        
        item_emb = getattr(self.sasrec, 'item_emb')
        item_embs = item_emb(torch.as_tensor(item_indices, dtype=torch.long, device=self.sasrec.dev))
        
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class AdaptiveGraphScaler(nn.Module):
    """
    Dynamic scale adjustment (0.2-0.4 range) based on training progress
    Helps with continual learning by adjusting graph influence
    """
    
    def __init__(
        self,
        initial_scale: float = 0.2,
        final_scale: float = 0.4,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.warmup_steps = warmup_steps
        self.register_buffer('step', torch.tensor(0))
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply dynamic scaling to embeddings"""
        if self.training:
            progress = min(1.0, float(self.step) / self.warmup_steps)
            scale = self.initial_scale + (self.final_scale - self.initial_scale) * progress
            self.step += 1
        else:
            scale = self.final_scale
        
        return embeddings * scale
    
    def reset_scale(self):
        """Reset for continual learning scenarios"""
        self.step.zero_()


def create_graph_enhanced_sasrec(
    sasrec_model: nn.Module,
    graph_embeddings: torch.Tensor,
    injection_mode: str = "gate",
    use_adaptive_scale: bool = True,
    initial_scale: float = 0.2,
    final_scale: float = 0.4,
) -> nn.Module:
    """
    Factory function to create SURGE-enhanced SASRec
    
    Args:
        sasrec_model: Base SASRec model
        graph_embeddings: Pre-trained KGLN embeddings [num_items+1, dim]
        injection_mode: How to inject graph embeddings (gate, concat, residual)
        use_adaptive_scale: Whether to use dynamic scaling
        initial_scale: Initial scaling factor
        final_scale: Final scaling factor after warmup
    
    Returns:
        Enhanced SASRec model with SURGE-style graph integration
    """
    if use_adaptive_scale:
        scaler = AdaptiveGraphScaler(initial_scale, final_scale)
        scaled_embeddings = scaler(graph_embeddings)
    else:
        scaled_embeddings = graph_embeddings * initial_scale
    
    model = GraphEnhancedSASRec(
        sasrec_model,
        scaled_embeddings,
        injection_mode=injection_mode,
        graph_weight=0.3,
        freeze_graph=True,
    )
    
    if use_adaptive_scale:
        model.scaler = scaler
    
    return model
