import torch
import torch.nn as nn
from benchmark_models._abstract_model import SequentialRecModel
from benchmark_models._modules import LayerNorm, TransformerEncoder

"""
[Paper]
Author: Fei Sun et al.
Title: "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
Conference: CIKM 2019

[Code Reference]
https://github.com/FeiSun/BERT4Rec
https://github.com/RUCAIBox/RecBole
"""

class BERT4RecModel(SequentialRecModel):
    def __init__(self, args):
        super(BERT4RecModel, self).__init__(args)

        # load parameters info
        self.mask_ratio = args.mask_ratio
        self.max_seq_length = args.max_seq_length
        self.item_embeddings = nn.Embedding(args.item_size+1, args.hidden_size, padding_idx=0)

        # load dataset info
        self.mask_token = args.item_size
        self.n_items = args.item_size
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_encoder = TransformerEncoder(args)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # parameters initialization
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_bi_attention_mask(input_ids)
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

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot
    
    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        """BERT4Rec Cloze-style MLM loss.

        We supervise only the `mask_num` randomly chosen positions per
        sequence (ignoring padding tokens). This is identical in both
        ``seq`` (1 sample/user) and ``prefix`` (1 target/prefix) dataloader
        modes — the shape of `answers` is not consumed here, BERT4Rec
        self-supervises via masking. Evaluating uses `predict()` which
        appends a mask token at the end.
        """
        B, L = input_ids.size()
        mask_num = max(1, int(L * self.mask_ratio))

        # Vectorised mask selection: rank uniform noise, keep top-k
        # positions. Avoids Python-level multinomial per sample.
        rand = torch.rand(B, L, device=input_ids.device)
        masked_index = rand.topk(mask_num, dim=1).indices  # (B, mask_num)

        pos_items = input_ids.gather(dim=1, index=masked_index)  # (B, mask_num)
        masked_input = input_ids.clone()
        masked_input.scatter_(
            dim=1, index=masked_index,
            src=torch.full_like(masked_index, self.mask_token),
        )

        seq_output = self.forward(masked_input)  # (B, L, H)

        # Gather hidden states at the masked positions only.
        idx_expand = masked_index.unsqueeze(-1).expand(-1, -1, seq_output.size(-1))
        masked_hidden = seq_output.gather(dim=1, index=idx_expand)  # (B, mask_num, H)

        test_item_emb = self.item_embeddings.weight[: self.n_items]  # (V, H)
        logits = torch.matmul(masked_hidden, test_item_emb.transpose(0, 1))  # (B, mask_num, V)

        # Ignore masks that landed on padding positions (pos_items == 0).
        valid = (pos_items > 0)
        if valid.any():
            loss = nn.CrossEntropyLoss()(
                logits[valid], pos_items[valid].long()
            )
        else:
            # Degenerate mini-batch where every mask hit padding: return a
            # zero loss attached to parameters so backward() is a no-op.
            loss = seq_output.sum() * 0.0

        return loss


    def predict(self, input_ids, user_ids, all_sequence_output=False):
        item_seq = self.reconstruct_test_data(input_ids)
        seq_output = self.forward(item_seq)

        return seq_output

    def reconstruct_test_data(self, item_seq):
        """
        Add mask token at the last position according to the lengths of item_seq
        """

        padding = self.mask_token * torch.ones(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        item_seq = item_seq[:, 1:]

        return item_seq