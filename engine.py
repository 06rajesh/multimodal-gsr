# ----------------------------------------------------------------------------------------------
# MGSRTR Official Code
# Copyright (c) Rajesh Baidya. All Rights Reserved
# ----------------------------------------------------------------------------------------------
# Modified from GSRTR (https://github.com/jhcho99/gsrtr)
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
from typing import List
import sys
import torch
import util.misc as utils
from typing import Iterable
from tqdm import tqdm

from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter
from models.types import ModelType
from models.verb_extractor import get_captions_from_tuple

def train_one_epoch(model: torch.nn.Module, tokenizer: BertTokenizer, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    model_type:ModelType = ModelType.MGSRTR, writer:SummaryWriter = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq = 100

    n_batches = len(data_loader)
    loader_desc = 'Epoch [{:d}]: lr = {:.4f}, loss = {:.4f}, accuracy (verb = {:.4f}, noun = {:.4f}, bounding box =  {:.4f})'
    train_iterator = tqdm(data_loader, desc=loader_desc.format(epoch, 0.0, 0.0, 0.0, 0.0, 0.0))

    for idx, (samples, captions, targets) in enumerate(train_iterator, 1):
        text_inputs = captions
        if model_type == ModelType.DuelEncGSR or model_type == ModelType.T5_MGSRTR:
            text_inputs = get_captions_from_tuple(captions)

        inputs = dict()

        if tokenizer:
            inputs = tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            mask = inputs['attention_mask'].bool()

            inputs.update({
                "mask": mask
            })

            inputs = inputs.to(device)

        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        # model output & calculate loss
        if model_type == ModelType.GSRTR:
            outputs = model(samples, targets)
        else:
            outputs = model(samples, inputs, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # stop when loss is nan or inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # loss backward & optimzer step
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if idx%print_freq == 0:
            items = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            train_iterator.set_description(loader_desc.format(epoch, items['lr'], items['loss'], items['verb_acc_top1_unscaled'],
                                            items['noun_acc_all_top1_unscaled'], items['bbox_acc_top5_unscaled']))

            if writer:
                writer.add_scalar("training loss", items['loss'], epoch*n_batches+idx)
                writer.add_scalars('noun_accuracy', {
                    "top-1": items['noun_acc_top1_unscaled'],
                    "top-5": items['noun_acc_top5_unscaled'],
                }, epoch * n_batches + idx)
                writer.add_scalars('verb_accuracy', {
                    "top-1": items['verb_acc_top1_unscaled'],
                    "top-5": items['verb_acc_top5_unscaled'],
                }, epoch * n_batches + idx)
                writer.add_scalars('bounding_box_accuracy', {
                    "top-1": items['bbox_acc_top1_unscaled'],
                    "top-5": items['bbox_acc_top5_unscaled'],
                }, epoch * n_batches + idx)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_swig(model, tokenizer, criterion, data_loader, device,
                  model_type:ModelType = ModelType.MGSRTR,
                  image_only:bool = False,
                  captions_only:bool = False,
                  ):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 10

    loader_desc = 'Test: loss = {:.4f}, accuracy (verb = {:.4f}, noun = {:.4f}, bounding box =  {:.4f})'
    test_iterator = tqdm(data_loader, desc=loader_desc.format(0.0, 0.0, 0.0, 0.0))

    for idx, (samples, captions, targets) in enumerate(test_iterator, 1):
        text_inputs = captions
        if model_type == ModelType.DuelEncGSR or model_type == ModelType.T5_MGSRTR:
            text_inputs = get_captions_from_tuple(captions)

        if image_only:
            text_inputs = ["" for _ in range(len(text_inputs))]

        inputs = dict()
        if tokenizer:
            inputs = tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            mask = inputs['attention_mask'].bool()
            inputs.update({
                "mask": mask
            })

            inputs = inputs.to(device)

        if captions_only:
            samples.tensors = torch.zeros(samples.tensors.size())

        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        # model output & calculate loss
        if model_type == ModelType.GSRTR:
            outputs = model(samples, targets)
        else:
            outputs = model(samples, inputs, targets)
        loss_dict = criterion(outputs, targets, eval=True)
        weight_dict = criterion.weight_dict



        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        if idx % print_freq == 0:
            items = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            test_iterator.set_description(
                loader_desc.format(items['loss'], items['verb_acc_top1_unscaled'],
                 items['noun_acc_all_top1_unscaled'], items['bbox_acc_top5_unscaled']))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


@torch.no_grad()
def run_swig_analysis(model, tokenizer, criterion, data_loader, device, model_type:ModelType = ModelType.MGSRTR):
    model.eval()
    criterion.eval()

    correct_verbs = {}
    incorrect_verbs = {}

    incorrect_nouns = {}
    incorrect_roles = {}

    loader_desc = 'Test:'
    test_iterator = tqdm(data_loader, desc=loader_desc.format(0.0, 0.0, 0.0, 0.0))

    for idx, (samples, captions, targets) in enumerate(test_iterator, 1):
        text_inputs = captions
        if model_type == ModelType.DuelEncGSR or model_type == ModelType.T5_MGSRTR:
            text_inputs = get_captions_from_tuple(captions)

        inputs = dict()
        if tokenizer:
            inputs = tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            mask = inputs['attention_mask'].bool()
            inputs.update({
                "mask": mask
            })

            inputs = inputs.to(device)

        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        batch_size = len(targets)

        # model output & calculate loss
        if model_type == ModelType.GSRTR:
            outputs = model(samples, targets)
        else:
            outputs = model(samples, inputs, targets)

        # top-1 & top 5 verb acc and calculate verb loss
        assert 'pred_verb' in outputs
        verb_pred_logits = outputs['pred_verb'].squeeze(1)
        gt_verbs = torch.stack([t['verbs'] for t in targets])

        _, pred = verb_pred_logits.topk(1, 1, True, True)
        pred = pred.t()
        correct = gt_verbs.view(1, -1).expand_as(pred)
        for i in range(batch_size):
            item = correct[i].item()
            if correct[i] == pred[i]:
                if item in correct_verbs:
                    correct_verbs[item] += 1
                else:
                    correct_verbs[item] = 1
            else:
                if item in incorrect_verbs:
                    incorrect_verbs[item] += 1
                else:
                    incorrect_verbs[item] = 1

        assert 'pred_noun' in outputs
        pred_noun = outputs['pred_noun']

        for i in range(batch_size):
            p, t = pred_noun[i], targets[i]
            roles = t['roles']
            num_roles = len(roles)
            role_targ = t['labels'][:num_roles]
            role_targ = role_targ.long()
            role_pred = p[:num_roles]

            _, pred = role_pred.topk(1, 1, True, True)
            pred_tile = pred.unsqueeze(2).tile(1, 1, role_targ.shape[1])
            # num_roles x target_num -> num_roles x maxk x target_num
            target_tile = role_targ.unsqueeze(1).tile(1, pred.shape[1], 1)
            # num_roles x maxk x target_num
            correct_tile = pred_tile.eq(target_tile)
            # num_roles x maxk x target_num -> num_roles x maxk -> maxk x num_roles
            correct = correct_tile.any(2).t()
            correct = correct.squeeze()

            incorrects = ((correct==False).nonzero()).squeeze(1).tolist()
            for j in incorrects:
                try:
                    role_id = roles[j].item()
                except ValueError:
                    continue

                if role_id in incorrect_roles:
                    incorrect_roles[role_id] += 1
                else:
                    incorrect_roles[role_id] = 1

                inc_labels = target_tile[j].squeeze(0).tolist()
                inc_set = set(inc_labels)
                for inc_id in inc_set:
                    if inc_id in incorrect_nouns:
                        incorrect_nouns[inc_id] += 1
                    else:
                        incorrect_nouns[inc_id] = 1

    return incorrect_verbs, incorrect_nouns, incorrect_roles, correct_verbs