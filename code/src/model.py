import torch
from typing import Any, Tuple

from components import SwiGLU, RotaryEmbedding, EncoderLayer, MHCLayer


class SASRec(torch.nn.Module):
    def __init__(self, user_num: int, item_num: int, args: Any) -> None:
        super(SASRec, self).__init__()

        self.user_num: int = user_num
        self.item_num: int = item_num
        self.dev: torch.device = args.device
        self.norm_first: bool = args.norm_first
        
        self.num_streams: int = getattr(args, 'num_streams', 4) 

        self.item_emb: torch.nn.Embedding = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        
        head_dim: int = args.hidden_units // args.num_heads
        self.rope: RotaryEmbedding = RotaryEmbedding(head_dim, max_seq_len=args.maxlen)

        self.emb_dropout: torch.nn.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms: torch.nn.ModuleList = torch.nn.ModuleList()
        self.attention_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.forward_layernorms: torch.nn.ModuleList = torch.nn.ModuleList()
        self.forward_layers: torch.nn.ModuleList = torch.nn.ModuleList()

        self.mhc_attn_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.mhc_ffn_layers: torch.nn.ModuleList = torch.nn.ModuleList()

        self.last_layernorm: torch.nn.LayerNorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                EncoderLayer(args.hidden_units, args.num_heads, args.dropout_rate)
            )
            self.mhc_attn_layers.append(
                MHCLayer(args.hidden_units, num_streams=self.num_streams)
            )

            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(
                SwiGLU(args.hidden_units, args.dropout_rate)
            )
            self.mhc_ffn_layers.append(
                MHCLayer(args.hidden_units, num_streams=self.num_streams)
            )
            
    def log2feats(self, log_seqs: Any) -> torch.Tensor:
        seqs: torch.Tensor = self.item_emb(torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        x_streams = seqs.unsqueeze(2).repeat(1, 1, self.num_streams, 1)

        for i in range(len(self.attention_layers)):
            
            def attn_wrapper(x: torch.Tensor) -> torch.Tensor:
                if self.norm_first:
                    x = self.attention_layernorms[i](x)

                return self.attention_layers[i](
                    x, 
                    attn_mask=None,
                    rotary_emb_fn=self.rope
                )
            
            x_streams = self.mhc_attn_layers[i](x_streams, attn_wrapper)

            def ffn_wrapper(x: torch.Tensor) -> torch.Tensor:
                if self.norm_first:
                    x = self.forward_layernorms[i](x)
                return self.forward_layers[i](x)

            x_streams = self.mhc_ffn_layers[i](x_streams, ffn_wrapper)

        final_seqs = torch.mean(x_streams, dim=2)

        log_feats: torch.Tensor = self.last_layernorm(final_seqs) 

        return log_feats
    
    def forward(self, user_ids: Any, log_seqs: Any, pos_seqs: Any, neg_seqs: Any) -> Tuple[torch.Tensor, torch.Tensor]:      
        log_feats: torch.Tensor = self.log2feats(log_seqs) 

        pos_embs: torch.Tensor = self.item_emb(torch.as_tensor(pos_seqs, dtype=torch.long, device=self.dev))
        neg_ids: torch.Tensor = torch.as_tensor(neg_seqs, dtype=torch.long, device=self.dev)
        neg_embs: torch.Tensor = self.item_emb(neg_ids)

        pos_logits: torch.Tensor = (log_feats * pos_embs).sum(dim=-1)

        if neg_embs.dim() == 3:
            neg_logits: torch.Tensor = (log_feats * neg_embs).sum(dim=-1)
        else:
            neg_logits = (log_feats.unsqueeze(-2) * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits
    
    def predict(self, user_ids: Any, log_seqs: Any, item_indices: Any) -> torch.Tensor:
        log_feats: torch.Tensor = self.log2feats(log_seqs) 

        final_feat: torch.Tensor = log_feats[:, -1, :] 
        
        item_embs: torch.Tensor = self.item_emb(torch.as_tensor(item_indices, dtype=torch.long, device=self.dev)) 
        
        logits: torch.Tensor = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits