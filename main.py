import torch
import numpy as np
import random
import json
import datetime
import time
import os
from pathlib import Path
from dotenv import load_dotenv

from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_dataset
import util.misc as utils
from util.analysis import idx_key_to_label
from models import build_model
from models.dual_encoder_gsr import build_dual_enc_model
from engine import train_one_epoch, evaluate_swig, run_swig_analysis
from dual_enc_engine import train_one_epoch_dual_enc, evaluate_flicker
from torch.utils.tensorboard import SummaryWriter
from models.types import Namespace, ModelType
from models.mgsrtr_config import MGSRTRConfig

def main(args:MGSRTRConfig, captions_only: bool = False, images_only: bool = False):
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    output_dir = Path(args.output_dir)

    summary_dir = output_dir / 'summary' / str(args.model_type.value)
    writer = SummaryWriter(str(summary_dir))

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # check dataset
    if args.dataset_file == "swig" or args.dataset_file == 'flicker30k':
        from datasets.swig import collater
    else:
        assert False, f"dataset {args.dataset_file} is not supported now"

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    args.num_noun_classes = dataset_train.num_nouns()
    if not args.test:
        dataset_val = build_dataset(image_set='val', args=args)
    else:
        dataset_test = build_dataset(image_set='test', args=args)

    # build model
    if args.model_type == ModelType.DuelEncGSR:
        model, tokenizer, criterion = build_dual_enc_model(args)
    elif args.model_type == ModelType.GSRTR:
        model, criterion = build_model(args)
        tokenizer = None
    else:
        model, tokenizer, criterion = build_model(args)

    model.to(device)
    model_without_ddp = model

    if args.resume:
        model_path = Path(args.output_dir, args.saved_model)
        if args.model_type == ModelType.DuelEncGSR:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
        else:
            model.soft_load_from_pretrained(str(model_path), device=device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    # optimizer & LR scheduler
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # dataset sampler
    if not args.test and not args.dev:
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        if args.dev:
            if args.distributed:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        elif args.test:
            if args.distributed:
                sampler_test = DistributedSampler(dataset_test, shuffle=False)
            else:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # dataset loader
    if not args.test and not args.dev:
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, num_workers=args.num_workers,
                                       collate_fn=collater, batch_sampler=batch_sampler_train)
        data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                                     drop_last=False, collate_fn=collater, sampler=sampler_val)
    else:
        if args.dev:
            data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                                         drop_last=False, collate_fn=collater, sampler=sampler_val)
        elif args.test:
            data_loader_test = DataLoader(dataset_test, num_workers=args.num_workers,
                                          drop_last=False, collate_fn=collater, sampler=sampler_test)

    # use saved model for evaluation (using dev set or test set)
    if args.dev or args.test:
        # checkpoint = torch.load(args.saved_model, map_location='cpu')
        # model.load_state_dict(checkpoint['model'])
        if args.dev:
            data_loader = data_loader_val
        elif args.test:
            data_loader = data_loader_test

        if args.analysis:
            log_stats = {}
            verbs, nouns, roles, _ = run_swig_analysis(model, tokenizer, criterion, data_loader, device, args.output_dir)
            verbs_stat = idx_key_to_label(verbs, args.idx_to_verb)
            log_stats['verbs'] = verbs_stat
            noun_stats = idx_key_to_label(nouns, args.idx_to_class)
            log_stats['nouns'] = noun_stats
            role_stats = idx_key_to_label(roles, args.idx_to_role)
            log_stats['roles'] = role_stats

            # write log
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log_stats.txt").open("w") as f:
                    f.write(json.dumps(log_stats) + "\n")

        else:
            if args.model_type == ModelType.DuelEncGSR:
                test_stats = evaluate_flicker(model, criterion, data_loader, device, args.output_dir)
            else:
                test_stats = evaluate_swig(model, tokenizer, criterion, data_loader, device, args.model_type, images_only=images_only, captions_only=captions_only)
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

            # write log
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log_tests.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        return None

    # train model
    print("Start training")
    start_time = time.time()
    max_test_mean_acc = 42

    # save config before start training
    args.save_config()

    for epoch in range(args.start_epoch, args.epochs):
        # train one epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)

        if args.model_type == ModelType.DuelEncGSR:
            train_stats = train_one_epoch_dual_enc(model, tokenizer, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, writer=writer)
        else:
            train_stats = train_one_epoch(model, tokenizer, criterion, data_loader_train, optimizer,
                                      device, epoch, max_norm=args.clip_max_norm, model_type=args.model_type, writer=writer)
        lr_scheduler.step()

        # evaluate
        if args.model_type == ModelType.DuelEncGSR:
            test_stats = evaluate_flicker(model, tokenizer, criterion, data_loader_val, device)
        else:
            test_stats = evaluate_swig(model, tokenizer, criterion, data_loader_val, device, model_type=args.model_type)

        # log & output
        # **{f'test_{k}': v for k, v in test_stats.items()},
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        writer.add_scalars('epoch_loss', {
            "training": train_stats['loss'],
            "validation": test_stats['loss'],
        }, epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # save checkpoint for every new max accuracy
            if log_stats['test_mean_acc_unscaled'] > max_test_mean_acc:
                max_test_mean_acc = log_stats['test_mean_acc_unscaled']
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'model': model_without_ddp.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'lr_scheduler': lr_scheduler.state_dict(),
                                      'epoch': epoch,
                                      'args': args}, checkpoint_path)
        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = MGSRTRConfig.from_env()

    # args = MGSRTRConfig.from_config('./flicker30k/pretrained/v7/config.json')
    args.test = True
    args.analysis = False

    main(args, images_only=True)