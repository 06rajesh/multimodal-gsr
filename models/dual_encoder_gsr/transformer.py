
import sys

# setting path
sys.path.append('../models')

import copy
import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor

from models.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from .t5_encoder import T5Encoder

class Transformer(nn.Module):

    def __init__(self, text_encoder, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_verb_classes = 504

        self.text_encoder = text_encoder
        self.text_encoder.requires_grad_(False)

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

        self.text_input_proj = nn.Linear(768, d_model)

        # classifer (for verb prediction)
        self.verb_classifier = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(d_model * 2, self.num_verb_classes))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, text_src, text_mask, enc_verb_query_embed, verb_query_embed, role_query_embed, pos_embed, text_pos_embed, vidx_ridx,
                targets=None, inference=False):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        text_pos_embed = torch.transpose(text_pos_embed, 0, 1)
        device = enc_verb_query_embed.device
        combined_pos_embed = torch.cat((text_pos_embed, pos_embed), dim=0)

        # Transformer Encoder (w/ verb classifier)
        enc_verb_query_embed = enc_verb_query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        zero_mask = torch.zeros((bs, 1), dtype=torch.bool, device=device)
        mem_mask = torch.cat([zero_mask, mask], dim=1)

        text_memory, attn_mask = self.text_encoder(input_ids=text_src, attention_mask=text_mask)
        text_memory = self.text_input_proj(text_memory)
        text_memory = torch.transpose(text_memory, 0, 1)
        text_mask = text_mask.bool()

        combined_mask = torch.cat((text_mask, mask), dim=1)

        verb_with_src = torch.cat([enc_verb_query_embed, src], dim=0)
        memory = self.encoder(verb_with_src, src_key_padding_mask=mem_mask, pos=pos_embed)
        vhs, memory = memory.split([1, h * w], dim=0)
        vhs = vhs.view(bs, -1)
        verb_pred = self.verb_classifier(vhs).view(bs, self.num_verb_classes)

        memory_combined = torch.cat((text_memory, memory), dim=0)

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
        rhs = self.decoder(role_tgt, memory_combined, memory_key_padding_mask=combined_mask,
                           pos=combined_pos_embed, query_pos=vr_query_embed)
        rhs = rhs.transpose(1, 2)

        return verb_pred, rhs, num_roles


def build_dual_enc_transformer(args):

    t5_encoder = T5Encoder(model_name='base', max_length=args.max_sentence_len)
    t5_encoder.setup()

    return Transformer(t5_encoder,
                       d_model=args.hidden_dim,
                       dropout=args.dropout,
                       nhead=args.nheads,
                       dim_feedforward=args.dim_feedforward,
                       num_encoder_layers=args.enc_layers,
                       num_decoder_layers=args.dec_layers)