import torch
from typing import Any, Tuple

from components import PointWiseFeedForward, RotaryEmbedding, EncoderLayer

class SASRec(torch.nn.Module):
    def __init__(self, user_num: int, item_num: int, args: Any) -> None:
        super(SASRec, self).__init__()

        self.user_num: int = user_num
        self.item_num: int = item_num
        self.dev: torch.device = args.device
        self.norm_first: bool = args.norm_first

        self.item_emb: torch.nn.Embedding = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        
        head_dim: int = args.hidden_units // args.num_heads
        self.rope: RotaryEmbedding = RotaryEmbedding(head_dim, max_seq_len=args.maxlen)

        self.emb_dropout: torch.nn.Dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms: torch.nn.ModuleList = torch.nn.ModuleList()
        self.attention_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.forward_layernorms: torch.nn.ModuleList = torch.nn.ModuleList()
        self.forward_layers: torch.nn.ModuleList = torch.nn.ModuleList()

        self.last_layernorm: torch.nn.LayerNorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm: torch.nn.LayerNorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer: EncoderLayer = EncoderLayer(
                args.hidden_units,
                args.num_heads,
                args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm: torch.nn.LayerNorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer: PointWiseFeedForward = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            
    def log2feats(self, log_seqs: Any) -> torch.Tensor:
        seqs: torch.Tensor = self.item_emb(torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        
        seqs = self.emb_dropout(seqs)

        tl: int = seqs.shape[1]
        attention_mask: torch.Tensor = torch.tril(torch.ones((tl, tl), device=self.dev))

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x: torch.Tensor = self.attention_layernorms[i](seqs)
                
                mha_outputs: torch.Tensor = self.attention_layers[i](
                    x, 
                    attn_mask=attention_mask, 
                    rotary_emb_fn=self.rope
                )
                
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            
            else:
                mha_outputs: torch.Tensor = self.attention_layers[i](
                    seqs, 
                    attn_mask=attention_mask, 
                    rotary_emb_fn=self.rope
                )
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats: torch.Tensor = self.last_layernorm(seqs) 

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