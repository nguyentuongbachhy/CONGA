import torch
from torch import nn
from benchmark_models._abstract_model import SequentialRecModel

"""
[Paper]
Author: Yong Kiam Tan et al.
Title: "Improved Recurrent Neural Networks for Session-based Recommendations."
Conference: DLRS 2016

[Code Reference]
https://github.com/RUCAIBox/RecBole
"""

class GRU4RecModel(SequentialRecModel):

    def __init__(self, args):
        super(GRU4RecModel, self).__init__(args)

        # load parameters info
        self.args = args
        self.embedding_size = args.hidden_size
        self.hidden_size = args.gru_hidden_size
        self.num_layers = args.num_hidden_layers
        self.dropout_prob = args.hidden_dropout_prob

        # define layers and loss
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # parameters initialization
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        item_seq_emb = self.item_embeddings(input_ids)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        return gru_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_out = self.forward(input_ids)  # (B, L, H) from GRU
        gamma = 1e-10

        if answers.dim() == 2:
            pos_ids, neg_ids = answers, neg_answers
            pos_emb = self.item_embeddings(pos_ids)
            neg_emb = self.item_embeddings(neg_ids)
            pos_logits = (pos_emb * seq_out).sum(-1)
            neg_logits = (neg_emb * seq_out).sum(-1)
            mask = (pos_ids != 0)
            loss = -torch.log(gamma + torch.sigmoid(pos_logits - neg_logits))
            loss = loss[mask].mean()
            return loss

        # Legacy prefix path.
        seq_last = seq_out[:, -1, :]
        pos_ids, neg_ids = answers, neg_answers
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)
        pos_logits = torch.sum(pos_emb * seq_last, -1)
        neg_logits = torch.sum(neg_emb * seq_last, -1)
        loss = -torch.log(gamma + torch.sigmoid(pos_logits - neg_logits)).mean()
        return loss
