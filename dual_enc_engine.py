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
import sys
import torch
import util.misc as utils
from typing import Iterable
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer
from frame_semantic_transformer.data.tasks import FrameClassificationTask


def train_one_epoch_dual_enc(model: torch.nn.Module, tokenizer:T5Tokenizer, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer:SummaryWriter = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq = 100

    n_batches = len(data_loader)
    loader_desc = 'Epoch [{:d}]: lr = {:.4f}, loss = {:.4f}, accuracy (verb = {:.4f}, noun = {:.4f}, bounding box =  {:.4f})'
    train_iterator = tqdm(data_loader, desc=loader_desc.format(epoch, 0.0, 0.0, 0.0, 0.0, 0.0))

    for idx, (samples, captions, targets) in enumerate(train_iterator, 1):

        tasks = []
        for c in captions:
            tasks.append(FrameClassificationTask(text=c[0], trigger_loc=c[1]))

        text_captions = [task.get_input() for task in tasks]

        inputs = tokenizer(
            text_captions,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # data & target
        samples = samples.to(device)
        inputs = inputs.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        # model output & calculate loss
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
                writer.add_scalars('accuracy', {
                    "noun": items['noun_acc_all_top1_unscaled'],
                    "verb": items['verb_acc_top1_unscaled'],
                    "bounding box": items['bbox_acc_top5_unscaled'],
                }, epoch*n_batches+idx)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_flicker(model, tokenizer:T5Tokenizer, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 10

    loader_desc = 'Test: loss = {:.4f}, accuracy (verb = {:.4f}, noun = {:.4f}, bounding box =  {:.4f})'
    test_iterator = tqdm(data_loader, desc=loader_desc.format(0.0, 0.0, 0.0, 0.0))

    for idx, (samples, captions, targets) in enumerate(test_iterator, 1):
        tasks = []
        for c in captions:
            tasks.append(FrameClassificationTask(text=c[0], trigger_loc=c[1]))

        captions = [task.get_input() for task in tasks]

        inputs = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # data & target
        samples = samples.to(device)
        inputs = inputs.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        # model output & calculate loss
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
