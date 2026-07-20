import torch
import torch.nn as nn
import copy
from benchmark_models._abstract_model import SequentialRecModel
from benchmark_models._modules import TransformerEncoder, LayerNorm

"""
[Paper]
Author: Wang-Cheng Kang et al. 
Title: "Self-Attentive Sequential Recommendation."
Conference: ICDM 2018

[Code Reference]
https://github.com/kang205/SASRec
https://github.com/Woeee/FMLP-Rec
"""

class SASRecModel(SequentialRecModel):
    def __init__(self, args):
        super(SASRecModel, self).__init__(args)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.item_encoder = TransformerEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        # Two paths, picked by the benchmark dataloader:
        #   - "seq"    batches: answers/neg_answers are (B, L) per-position
        #     next-item targets (SASRec paper protocol, ~maxlen x faster).
        #   - "prefix" batches: answers/neg_answers are scalars per sample.
        seq_out = self.forward(input_ids)  # (B, L, H)

        if answers.dim() == 2:
            pos_ids, neg_ids = answers, neg_answers
            pos_emb = self.item_embeddings(pos_ids)
            neg_emb = self.item_embeddings(neg_ids)
            pos_logits = (pos_emb * seq_out).sum(-1)
            neg_logits = (neg_emb * seq_out).sum(-1)
            mask = (pos_ids != 0)
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            bce = torch.nn.BCEWithLogitsLoss(reduction="none")
            loss = (bce(pos_logits, pos_labels) + bce(neg_logits, neg_labels))
            loss = loss[mask].mean()
            return loss

        # Legacy prefix path (last-position supervision).
        seq_last = seq_out[:, -1, :]
        pos_ids, neg_ids = answers, neg_answers
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)
        pos_logits = torch.sum(pos_emb * seq_last, -1)
        neg_logits = torch.sum(neg_emb * seq_last, -1)
        pos_labels = torch.ones(pos_logits.shape, device=seq_last.device)
        neg_labels = torch.zeros(neg_logits.shape, device=seq_last.device)
        indices = (pos_ids != 0).nonzero().reshape(-1)
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        return loss
