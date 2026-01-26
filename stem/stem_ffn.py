import torch
from typing import Any, Tuple

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
    STEM-enhanced SwiGLU: Replaces up-projection (w1) with item-indexed embeddings.
    
    Standard SwiGLU:
        w1_out = W1 @ x          # [hidden, hidden*4] @ [batch, seq, hidden]
        w2_out = W2 @ x          # [hidden, hidden*4] @ [batch, seq, hidden]
        hidden = SwiGLU(w1_out, w2_out)
        output = W3 @ hidden     # [hidden*4, hidden] @ [batch, seq, hidden*4]
    
    STEM SwiGLU:
        w1_out = U[item_ids]     # [batch, seq, hidden*4] - item-indexed lookup
        w2_out = W2 @ x          # [hidden, hidden*4] @ [batch, seq, hidden]
        hidden = SwiGLU(w1_out, w2_out)
        output = W3 @ hidden     # [hidden*4, hidden] @ [batch, seq, hidden*4]
    
    Benefits:
    - 33% fewer FFN parameters (W1 removed)
    - Item-specific knowledge storage
    - Better interpretability
    - Stable training (no routing like MoE)
    """
    
    def __init__(
        self, 
        hidden_units: int, 
        num_items: int,
        dropout_rate: float,
        use_stem: bool = True,
        padding_idx: int = 0
    ) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.num_items = num_items
        self.use_stem = use_stem
        self.ffn_dim = hidden_units * 4
        
        if use_stem:
            # STEM: Replace w1 with item-indexed embeddings
            self.item_up_embeddings = torch.nn.Embedding(
                num_items, 
                self.ffn_dim,
                padding_idx=padding_idx
            )
            # Initialize with small values for stability
            torch.nn.init.normal_(self.item_up_embeddings.weight, mean=0.0, std=0.02)
            if padding_idx is not None:
                with torch.no_grad():
                    self.item_up_embeddings.weight[padding_idx].fill_(0)
        else:
            # Standard: Dense up-projection
            self.w1 = torch.nn.Linear(hidden_units, self.ffn_dim, bias=False)
        
        # Gate and down-projection remain unchanged
        self.w2 = torch.nn.Linear(hidden_units, self.ffn_dim, bias=False)
        self.w3 = torch.nn.Linear(self.ffn_dim, hidden_units, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, item_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Hidden states [batch_size, seq_len, hidden_units]
            item_ids: Item indices [batch_size, seq_len] (required if use_stem=True)
        
        Returns:
            Output [batch_size, seq_len, hidden_units]
        """
        if self.use_stem:
            if item_ids is None:
                raise ValueError("item_ids required when use_stem=True")
            # STEM: Lookup item-specific up-projection
            w1_out = self.item_up_embeddings(item_ids)
        else:
            # Standard: Dense up-projection
            w1_out = self.w1(x)
        
        # Gate projection (unchanged)
        w2_out = self.w2(x)
        
        # SwiGLU activation
        hidden = STEMSwiGLUFunction.apply(w1_out, w2_out)
        
        # Down-projection
        return self.w3(self.dropout(hidden))
    
    def init_from_pretrained(self, pretrained_embeddings: torch.Tensor):
        """
        Initialize STEM embeddings from pretrained item embeddings (e.g., from graph).
        
        Args:
            pretrained_embeddings: [num_items, embedding_dim]
        """
        if not self.use_stem:
            raise ValueError("Cannot init pretrained embeddings when use_stem=False")
        
        with torch.no_grad():
            # Project pretrained embeddings to FFN dimension if needed
            if pretrained_embeddings.size(1) != self.ffn_dim:
                # Simple linear projection
                projection = torch.nn.Linear(
                    pretrained_embeddings.size(1), 
                    self.ffn_dim,
                    bias=False
                ).to(pretrained_embeddings.device)
                torch.nn.init.xavier_normal_(projection.weight)
                projected = projection(pretrained_embeddings)
                self.item_up_embeddings.weight.copy_(projected)
            else:
                self.item_up_embeddings.weight.copy_(pretrained_embeddings)
            
            # Zero out padding
            if self.item_up_embeddings.padding_idx is not None:
                self.item_up_embeddings.weight[self.item_up_embeddings.padding_idx].fill_(0)
