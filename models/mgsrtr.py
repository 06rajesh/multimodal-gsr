# ----------------------------------------------------------------------------------------------
# MGSRTR Official Code
# Copyright (c) Rajesh Baidya. All Rights Reserved
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from GSRTR (https://github.com/jhcho99/gsrtr)
# Copyright (c) Junhyeong Cho (jhcho99.cs@gmail.com) [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
MGSRTR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_swig, accuracy_swig_bbox)

from transformers import BertTokenizer, VisualBertForVisualReasoning, VisualBertConfig, BertModel
from torchsummary import summary

from .visual_bert import VisualBertEmbeddings
from .multi_transformer import MultiTransformer
from .backbone import Backbone
from .gsrtr import SWiGCriterion

class MGSRTR(nn.Module):
    """ GSRTR model for Grounded Situation Recognition"""

    def __init__(self, backbone, embeddings, transformer, num_noun_classes, vidx_ridx, num_roles=190, num_verbs=504):
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer architecture. See transformer.py
            - num_noun_classes: the number of noun classes
            - vidx_ridx: verb index to role index
        """
        super().__init__()
        self.backbone = backbone
        self.embeddings = embeddings
        self.transformer = transformer
        self.num_noun_classes = num_noun_classes
        self.vidx_ridx = vidx_ridx
        self.num_role_queries = num_roles
        self.num_verb_queries = num_verbs

        # hidden dimension for queries and image features
        hidden_dim = transformer.d_model

        # query embeddings
        self.role_query_embed = nn.Embedding(self.num_role_queries, hidden_dim // 2)
        self.verb_query_embed = nn.Embedding(self.num_verb_queries, hidden_dim // 2)
        self.enc_verb_query_embed = nn.Embedding(1, hidden_dim)

        # 1x1 Conv
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

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

        features = self.backbone(samples)
        f_key = list(features.keys())[-1]
        src, mask = features[f_key].decompose()

        assert mask is not None
        assert text_inputs.mask is not None

        projected = src.flatten(2)
        projected = torch.transpose(projected, 1, 2)

        embedding, pos = self.embeddings(
            input_ids=text_inputs.input_ids,
            token_type_ids=text_inputs.token_type_ids,
            attention_mask=text_inputs.attention_mask,
            visual_embeds=projected,
            visual_token_type_ids=None,
        )

        embedding = embedding.permute(0, 2, 1)
        pos = pos.permute(0, 2, 1)

        mask = mask.flatten(1)
        combined_mask = torch.cat([text_inputs.mask, mask], 1)

        batch_size = src.shape[0]
        batch_verb, batch_noun, batch_bbox, batch_bbox_conf = [], [], [], []

        # model prediction
        for i in range(batch_size):
            if not inference:
                verb_pred, rhs, num_roles = self.transformer(embedding[i:i+1],
                                                             combined_mask[i:i + 1], self.enc_verb_query_embed.weight,
                                                             self.verb_query_embed.weight, self.role_query_embed.weight,
                                                             pos[i:i+1], self.vidx_ridx, targets=targets[i],
                                                             inference=inference)
            else:
                verb_pred, rhs, num_roles = self.transformer(embedding[i:i+1],
                                                             combined_mask[i:i + 1], self.enc_verb_query_embed.weight,
                                                             self.verb_query_embed.weight, self.role_query_embed.weight,
                                                             pos[i:i+1], self.vidx_ridx, inference=inference)
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
    if not args.inference:
        train_backbone = args.lr_backbone > 0
    else:
        train_backbone = False
    backbone = Backbone(args.backbone, train_backbone, False, False)

    transformer = MultiTransformer(d_model=args.hidden_dim,
                                   dropout=args.dropout,
                                   nhead=args.nheads,
                                   dim_feedforward=args.dim_feedforward,
                                   num_encoder_layers=args.enc_layers,
                                   num_decoder_layers=args.dec_layers)

    bertmodelname = "bert-base-uncased"
    # bertmodel = BertModel.from_pretrained(bertmodelname)
    # embedding_matrix = bertmodel.embeddings.word_embeddings.weight

    tokenizer = BertTokenizer.from_pretrained(bertmodelname, model_max_length=args.max_sentence_len)

    vbconfig=VisualBertConfig()
    vbconfig.visual_embedding_dim = args.dim_feedforward
    vbconfig.word_embedding_dim = vbconfig.hidden_size # embedding_matrix.shape[1]
    vbconfig.hidden_size = args.hidden_dim
    vbconfig.batch_size = args.batch_size

    vbertembedding = VisualBertEmbeddings(vbconfig, bert_model_name=bertmodelname)

    model = MGSRTR(backbone,
                   vbertembedding,
                   transformer,
                   num_noun_classes=args.num_noun_classes,
                   vidx_ridx=args.vidx_ridx,
                   num_roles=len(args.idx_to_role),
                   num_verbs=len(args.idx_to_verb))
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
