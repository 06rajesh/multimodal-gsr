
import copy
import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor

from .transformer import  TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

class MultiTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_verb_classes = 504

        # encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # classifer (for verb prediction)
        self.verb_classifier = nn.Sequential(nn.Linear(d_model, d_model *2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(d_model *2, self.num_verb_classes))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, enc_verb_query_embed, verb_query_embed, role_query_embed, pos_embed, vidx_ridx, targets=None, inference=False):
        # flatten NxCxHxW to HWxNxC
        bs, c, h_w = src.shape
        device = enc_verb_query_embed.device

        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)

        # Transformer Encoder (w/ verb classifier)
        enc_verb_query_embed = enc_verb_query_embed.unsqueeze(1).repeat(1, bs, 1)
        zero_mask = torch.zeros((bs, 1), dtype=torch.bool, device=device)
        mem_mask = torch.cat([zero_mask, mask], dim=1)
        verb_with_src = torch.cat([enc_verb_query_embed, src], dim=0)
        memory = self.encoder(verb_with_src, src_key_padding_mask=mem_mask, pos=pos_embed)
        vhs, memory = memory.split([1, h_w], dim=0)
        vhs = vhs.view(bs, -1)
        verb_pred = self.verb_classifier(vhs).view(bs, self.num_verb_classes)

        # Transformer Decoder
        ## At training time, we assume that the ground-truth verb is given.
        #### Please see the evaluation details in [Grounded Situation Recognition] task.
        #### There are three evaluation settings: top-1 predicted verb, top-5 predicted verbs and ground-truth verb.
        #### If top-1 predicted verb is incorrect, then grounded noun predictions in top-1 predicted verb setting are considered incorrect.
        #### If the ground-truth verb is not included in top-5 predicted verbs, then grounded noun predictions in top-5 predicted verbs setting are considered incorrect.
        #### In ground-truth verb setting, we only consider grounded noun predictions.
        ## At inference time, we use the predicted verb.
        #### For semantic role queries, we select the verb query embedding corresponding to the predicted verb.
        #### For semantic role queries, we select the role query embeddings for the semantic roles associated with the predicted verb.

        if not inference:
            selected_verb_embed = verb_query_embed[targets['verbs']]
            selected_roles = targets['roles']
        else:
            top1_verb = torch.topk(verb_pred, k=1, dim=1)[1].item()
            selected_verb_embed = verb_query_embed[top1_verb]
            selected_roles = vidx_ridx[top1_verb]
        selected_query_embed = role_query_embed[selected_roles]
        num_roles = len(selected_roles)
        vr_query_embed = torch.cat([selected_query_embed,
                                    selected_verb_embed[None].tile(num_roles, 1)],
                                   axis=-1)
        vr_query_embed = vr_query_embed.unsqueeze(1).repeat(1, bs, 1)
        role_tgt = torch.zeros_like(vr_query_embed)
        rhs = self.decoder(role_tgt, memory, memory_key_padding_mask=mask,
                           pos=pos_embed, query_pos=vr_query_embed)
        rhs = rhs.transpose(1, 2)

        return verb_pred, rhs, num_roles