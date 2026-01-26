import torch
import torch.nn.functional as F
from typing import Any, Tuple, Optional

import swiglu_cuda


class STEMSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, w1_out: torch.Tensor, w2_out: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(w1_out, w2_out)
        return swiglu_cuda.fwd(w1_out, w2_out)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w1_out, w2_out = ctx.saved_tensors
        grad_w1, grad_w2 = swiglu_cuda.bwd(grad_out.contiguous(), w1_out, w2_out)
        return grad_w1, grad_w2


class STEMSwiGLU(torch.nn.Module):
    """
    STEM-enhanced SwiGLU: Replaces up-projection (w1) with token-indexed embeddings.
    Paper: "Scaling Transformers with Embedding Modules" (arXiv:2601.10639)
    
    Standard SwiGLU:
        w1_out = W1 @ x          # Dense up-projection
        w2_out = W2 @ x          # Gate projection
        hidden = SwiGLU(w1_out, w2_out)
        output = W3 @ hidden     # Down-projection
    
    STEM SwiGLU:
        w1_out = U[token_ids]    # Token-indexed lookup (layer-local)
        w2_out = W2 @ x          # Gate projection (context-aware)
        hidden = SwiGLU(w1_out, w2_out)
        output = W3 @ hidden     # Down-projection
    
    Benefits (Paper Results):
    - 33% fewer FFN parameters (W1 removed)
    - Token-specific knowledge storage (Luniq grows sublinear)
    - Better training stability (NO loss spikes vs MoE)
    - +9-10% on knowledge tasks (ARC-Challenge)
    - CPU offloading support for memory efficiency
    """
    
    def __init__(
        self, 
        hidden_units: int, 
        num_items: int,
        dropout_rate: float,
        use_stem: bool = True,
        padding_idx: int = 0,
        cpu_offload: bool = False
    ) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.num_items = num_items
        self.use_stem = use_stem
        self.cpu_offload = cpu_offload
        self.ffn_dim = hidden_units * 4
        self.padding_idx = padding_idx
        
        # Prefetch stream for async CPU->GPU transfer
        self.prefetch_stream = torch.cuda.Stream() if cpu_offload and torch.cuda.is_available() else None
        
        if use_stem:
            # STEM: Token-indexed embedding table (layer-local)
            device = 'cpu' if cpu_offload else 'cuda'
            self.token_embeddings = torch.nn.Embedding(
                num_items, 
                self.ffn_dim,
                padding_idx=padding_idx
            )
            
            # Initialize with small values (paper: std=0.02)
            torch.nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
            if padding_idx is not None:
                with torch.no_grad():
                    self.token_embeddings.weight[padding_idx].fill_(0)
            
            # Move to CPU if offloading enabled
            if cpu_offload:
                self.token_embeddings = self.token_embeddings.to('cpu')
        else:
            # Standard: Dense up-projection
            self.w1 = torch.nn.Linear(hidden_units, self.ffn_dim, bias=False)
        
        # Gate and down-projection (always on GPU, shared across tokens)
        self.w2 = torch.nn.Linear(hidden_units, self.ffn_dim, bias=False)
        self.w3 = torch.nn.Linear(self.ffn_dim, hidden_units, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Hidden states [batch_size, seq_len, hidden_units]
            token_ids: Token indices [batch_size, seq_len] (required if use_stem=True)
        
        Returns:
            Output [batch_size, seq_len, hidden_units]
        """
        if self.use_stem:
            if token_ids is None:
                raise ValueError("token_ids required when use_stem=True")
            
            # STEM: Token-indexed lookup with optional CPU offloading
            if self.cpu_offload and self.prefetch_stream is not None:
                # Async prefetch from CPU to GPU
                with torch.cuda.stream(self.prefetch_stream):
                    token_ids_cpu = token_ids.cpu()
                    w1_out = self.token_embeddings(token_ids_cpu).to(x.device, non_blocking=True)
                # Record stream for synchronization
                w1_out.record_stream(self.prefetch_stream)
            else:
                # Direct lookup (GPU or CPU)
                w1_out = self.token_embeddings(token_ids)
                if self.cpu_offload:
                    w1_out = w1_out.to(x.device)
        else:
            # Standard: Dense up-projection
            w1_out = self.w1(x)
        
        # Gate projection (context-aware modulation)
        w2_out = self.w2(x)
        
        # SwiGLU activation: element-wise gate * up
        hidden = STEMSwiGLUFunction.apply(w1_out, w2_out)
        
        # Down-projection
        return self.w3(self.dropout(hidden))
    
    def init_from_pretrained(self, pretrained_embeddings: torch.Tensor):
        """
        Initialize STEM embeddings from pretrained embeddings (e.g., graph, CMS).
        Paper Sec3.2: Pretrained init improves convergence.
        
        Args:
            pretrained_embeddings: [num_items, embedding_dim]
        """
        if not self.use_stem:
            raise ValueError("Cannot init pretrained embeddings when use_stem=False")
        
        with torch.no_grad():
            # Project to FFN dimension if needed
            if pretrained_embeddings.size(1) != self.ffn_dim:
                projection = torch.nn.Linear(
                    pretrained_embeddings.size(1), 
                    self.ffn_dim,
                    bias=False
                ).to(pretrained_embeddings.device)
                torch.nn.init.xavier_normal_(projection.weight)
                projected = projection(pretrained_embeddings)
                self.token_embeddings.weight.copy_(projected)
            else:
                self.token_embeddings.weight.copy_(pretrained_embeddings)
            
            # Zero out padding
            if self.padding_idx is not None:
                self.token_embeddings.weight[self.padding_idx].fill_(0)
    
    def get_param_count(self) -> dict:
        """Return parameter counts for analysis."""
        if self.use_stem:
            stem_params = self.token_embeddings.weight.numel()
            gate_params = self.w2.weight.numel()
            down_params = self.w3.weight.numel()
            total = stem_params + gate_params + down_params
            return {
                'stem_embeddings': stem_params,
                'gate': gate_params,
                'down': down_params,
                'total': total,
                'reduction_vs_dense': f"-{100 * (self.hidden_units * self.ffn_dim) / total:.1f}%"
            }
        else:
            up_params = self.w1.weight.numel()
            gate_params = self.w2.weight.numel()
            down_params = self.w3.weight.numel()
            return {
                'up': up_params,
                'gate': gate_params,
                'down': down_params,
                'total': up_params + gate_params + down_params
            }
