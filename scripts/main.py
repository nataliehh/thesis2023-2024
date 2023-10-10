import glob
import logging
import os
import re
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import GradScaler

from open_clip import create_model_and_transforms, get_tokenizer, create_loss
from .scheduler import cosine_lr, const_lr, const_lr_cooldown

import fsspec

# use custom functions
from .params import parse_args
from .data_loader import get_data
from .model import create_custom_model
from .train import train_one_epoch
from .evaluate import evaluate
from .loss import create_loss


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

def pt_load(file_path, map_location=None):
    if file_path.startswith('s3'):
        print('Loading remote checkpoint, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str):
    checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    # args = parse_args(args)

    if torch.cuda.is_available():
        # Enable tf32 on Ampere GPUs that's only 8% slower than float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    # get the name of the experiments
    if args.name is None:
        keyword_type = args.keyword_path.split('/')[-1].split('.')[0]\
            if args.keyword_path is not None else 'none'
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = '-'.join([
            date_str,
            f"data_{args.train_data}",
            f"ratio_{args.label_ratio}",
            f"model_{args.model}",
            f"method_{args.method}",
            f"keyword_{keyword_type}",
            f"seed_{args.seed}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if args.train_data:
        print('Making log path:', log_base_path)
        os.makedirs(log_base_path, exist_ok=True)
        os.makedirs(os.path.join(log_base_path, 'checkpoints'), exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from


    if args.precision == 'fp16':
        logging.warning('It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    print(f'Running with a single process. Device {device}.')

    dist_model = None

    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    log_base_path = os.path.join(args.logs, args.name)

    # if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
    #     # arg is nargs, single (square) image size list -> int
    #     args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model, args.pretrained, precision=args.precision, device=device, output_dict=True,)

    model = create_custom_model(args, model)  # use custom model

    random_seed(args.seed, args.rank)

    
    if args.train_data:
        # print("Model:")
        # print(f"{str(model)}")
        # print("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                # print(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data:
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [{"params": gain_or_bias_params, "weight_decay": 0.},
             {"params": rest_params, "weight_decay": args.wd},],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            print(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            print(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    print('Getting data...')
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    assert len(data), 'At least one train or eval dataset must be specified.'
    print('Data got.')
    # create scheduler if train
    scheduler = None
    if args.train_data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.train_data and args.logs and args.logs.lower() != 'none'

    writer = None
    if not args.train_data:
        metrics = evaluate(model, data, start_epoch, args, writer)
        with open('eval.txt', 'a') as f:
            for k, v in metrics.items():
                if k == "zeroshot-val-top1":
                    f.write('{}\t{}\t{}\t{:.2f}\n'.format(
                        args.name, args.imagenet_val, k, 100 * v))
                elif k in ["image_to_text_R@1", "image_to_text_R@5", "image_to_text_R@10",
                           "text_to_image_R@1", "text_to_image_R@5", "text_to_image_R@10"]:
                    f.write('{}\t{}\t{}\t{:.2f}\n'.format(
                        args.name, args.val_data, k, 100 * v))
        return

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        print(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints.
        checkpoint_dict = {"epoch": completed_epoch, "name": args.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        if scaler is not None:
            checkpoint_dict["scaler"] = scaler.state_dict()

        # only save the last epoch to save server storage
        if completed_epoch == args.epochs:
            torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_latest.pt"))

if __name__ == "__main__":
    main(sys.argv[1:])