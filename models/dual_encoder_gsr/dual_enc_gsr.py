# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

import sys

# setting path
sys.path.append('../models')

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_swig, accuracy_swig_bbox)
from models.backbone import build_backbone
from .transformer import build_dual_enc_transformer
from models.gsrtr import SWiGCriterion


class DualEncGSR(nn.Module):
    """ GSRTR model for Grounded Situation Recognition"""

    def __init__(self, backbone, transformer, max_sentence_length, batch_size, num_noun_classes, vidx_ridx):
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer architecture. See transformer.py
            - num_noun_classes: the number of noun classes
            - vidx_ridx: verb index to role index
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_noun_classes = num_noun_classes
        self.vidx_ridx = vidx_ridx
        self.num_role_queries = 190
        self.num_verb_queries = 504

        # hidden dimension for queries and image features
        hidden_dim = transformer.d_model

        self.text_pos_embed = nn.Embedding(max_sentence_length, hidden_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids",
                             torch.arange(100)
                             .expand((1, -1)).repeat(batch_size, 1)
                             )

        # query embeddings
        self.role_query_embed = nn.Embedding(self.num_role_queries, hidden_dim // 2)
        self.verb_query_embed = nn.Embedding(self.num_verb_queries, hidden_dim // 2)
        self.enc_verb_query_embed = nn.Embedding(1, hidden_dim)

        # 1x1 Conv
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # classifiers & predictors (for grounded noun prediction)
        self.noun_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(hidden_dim * 2, self.num_noun_classes))
        self.bbox_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(hidden_dim * 2, 4))
        self.bbox_conf_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.2),
                                                 nn.Linear(hidden_dim * 2, 1))

        print("GSRTR: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, samples, text_inputs, targets=None, inference=False):
        """ 
        Parameters:
               - samples: The forward expects a NestedTensor, which consists of:
                        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - targets: This has verbs, roles and labels information
               - inference: boolean, used in inference
        Outputs:
               - out: dict of tensors. 'pred_verb', 'pred_noun', 'pred_bbox' and 'pred_bbox_conf' are keys
        """
        MAX_NUM_ROLES = 6
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        input_shape = text_inputs['input_ids'].shape
        n_inputs = input_shape[0]
        seq_length = input_shape[1]

        position_ids = self.position_ids[:n_inputs, :seq_length]
        text_pos = self.text_pos_embed(position_ids)

        batch_size = src.shape[0]
        batch_verb, batch_noun, batch_bbox, batch_bbox_conf = [], [], [], []

        i = 0

        # model prediction
        for i in range(batch_size):
            if not inference:
                verb_pred, rhs, num_roles = self.transformer(self.input_proj(src[i:i + 1]), mask[i:i + 1],
                                                             text_inputs['input_ids'][i:i + 1], text_inputs['attention_mask'][i:i + 1],
                                                             self.enc_verb_query_embed.weight,
                                                             self.verb_query_embed.weight, self.role_query_embed.weight,
                                                             pos[-1][i:i + 1], text_pos[i:i + 1], self.vidx_ridx,
                                                             targets=targets[i], inference=inference)
            else:
                verb_pred, rhs, num_roles = self.transformer(self.input_proj(src[i:i + 1]), mask[i:i + 1],
                                                             text_inputs['input_ids'][i:i + 1], text_inputs['attention_mask'][i:i + 1],
                                                             self.enc_verb_query_embed.weight,
                                                             self.verb_query_embed.weight, self.role_query_embed.weight,
                                                             pos[-1][i:i + 1], text_pos[i:i + 1], self.vidx_ridx,
                                                             inference=inference)
            noun_pred = self.noun_classifier(rhs)
            noun_pred = F.pad(noun_pred, (0, 0, 0, MAX_NUM_ROLES - num_roles), mode='constant', value=0)[-1].view(1,
                                                                                                                  MAX_NUM_ROLES,
                                                                                                                  self.num_noun_classes)
            bbox_pred = self.bbox_predictor(rhs).sigmoid()
            bbox_pred = F.pad(bbox_pred, (0, 0, 0, MAX_NUM_ROLES - num_roles), mode='constant', value=0)[-1].view(1,
                                                                                                                  MAX_NUM_ROLES,
                                                                                                                  4)
            bbox_conf_pred = self.bbox_conf_predictor(rhs)
            bbox_conf_pred = F.pad(bbox_conf_pred, (0, 0, 0, MAX_NUM_ROLES - num_roles), mode='constant', value=0)[
                -1].view(1, MAX_NUM_ROLES, 1)

            batch_verb.append(verb_pred)
            batch_noun.append(noun_pred)
            batch_bbox.append(bbox_pred)
            batch_bbox_conf.append(bbox_conf_pred)

        # outputs
        out = {}
        out['pred_verb'] = torch.cat(batch_verb, dim=0)
        out['pred_noun'] = torch.cat(batch_noun, dim=0)
        out['pred_bbox'] = torch.cat(batch_bbox, dim=0)
        out['pred_bbox_conf'] = torch.cat(batch_bbox_conf, dim=0)

        return out


def build(args):
    backbone = build_backbone(args)
    transformer, tokenizer = build_dual_enc_transformer(args)

    model = DualEncGSR(
                backbone,
                transformer,
                max_sentence_length=args.max_sentence_len,
                batch_size=args.batch_size,
                num_noun_classes=args.num_noun_classes,
                vidx_ridx=args.vidx_ridx
            )

    criterion = None

    if not args.inference:
        weight_dict = {'loss_nce': args.noun_loss_coef, 'loss_vce': args.verb_loss_coef,
                       'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef,
                       'loss_bbox_conf': args.bbox_conf_loss_coef}

        if not args.test:
            criterion = SWiGCriterion(weight_dict=weight_dict,
                                      SWiG_json_train=args.SWiG_json_train,
                                      SWiG_json_eval=args.SWiG_json_dev,
                                      idx_to_role=args.idx_to_role)
        else:
            criterion = SWiGCriterion(weight_dict=weight_dict,
                                      SWiG_json_train=args.SWiG_json_train,
                                      SWiG_json_eval=args.SWiG_json_test,
                                      idx_to_role=args.idx_to_role)

    return model, tokenizer, criterion