import glob
import logging
import os
import re
import sys
import random
from datetime import datetime
import numpy as np
import torch
import fsspec
from open_clip import create_model_and_transforms, get_tokenizer, create_loss
from tqdm import tqdm
import gc

sys.path.append('/vol/tensusers4/nhollain/thesis2023-2024/s_clip_scripts') # Add custom functions to PATH
sys.path.append('/vol/tensusers4/nhollain/ProbVLM/src') # Allow probvlm imports

# use custom functions
from scheduler import cosine_lr, const_lr, const_lr_cooldown
from params import parse_args
from data_loader import get_data
from model import create_custom_model
from train import train_one_epoch
from evaluate import evaluate
from loss import create_loss

# ProbVLM imports
from networks import get_default_BayesCap_for_CLIP
from losses import TempCombLoss
from train_probVLM import train_ProbVLM

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

def get_optimizer_scaler(args, model):
    optimizer, scaler = None, None
    if args.train_data:
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [{"params": gain_or_bias_params, "weight_decay": 0.},
             {"params": rest_params, "weight_decay": args.wd},],
            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
        )
        scaler = torch.cuda.amp.GradScaler() if args.precision == "amp" else None
    return optimizer, scaler

# From: https://stackoverflow.com/questions/658763/how-to-suppress-scientific-notation-when-printing-float-values
def format_float(num): 
    return np.format_float_positional(num, trim='-')

def format_checkpoint(args):
    keyword_type = args.keyword_path.split('/')[-1].split('.')[0]\
        if args.keyword_path is not None else 'none'
    keyword_type = keyword_type.replace('-', '')
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    name = '-'.join([date_str, f"data_{args.train_data}",
        f"ratio_{args.label_ratio}", f"model_{args.model}", f"method_{args.method}", f"kw_{keyword_type}",
        f"ProbVLM_{args.probvlm}", f"AL.iter_{args.al_iter}", f"AL.epochs_{args.al_epochs}",
        f"PL_{args.pl_method}", f"vit_{args.use_vit}", f"epochs_{args.epochs}", "lr_" + format_float(args.lr), f"bs_{args.batch_size}",
        ])
    if args.k_fold >= 0:
        # print('kfold:', args.k_fold)
        name += f"-fold_{args.k_fold}"
    return name
    
def main(args):
    logging.basicConfig(filename='training.log', encoding='utf-8', level=logging.INFO, force=True) # filemode='a', 
    try: # If the arguments are not parsed yet, do it here
        args = parse_args(args)
        print('Parsed arguments.')
    except: # Otherwise, continue
        pass

    logging.info(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    

    if args.name is None: # get the name of the experiments
        # The keyword is the last part of the keyword filepath, except for its suffix
        args.name = format_checkpoint(args)

    resume_latest = args.resume == 'latest'
    
    log_base_path = os.path.join(args.logs, args.name)
    if args.train_data:
        logging.info('Log path: ' + log_base_path)
        os.makedirs(log_base_path, exist_ok=True)
        os.makedirs(os.path.join(log_base_path, 'checkpoints'), exist_ok=True)
        log_filename = 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print("Error. Experiment already exists. Use --name {} to specify a new experiment.")
            return -1

    if resume_latest:
        resume_from = None

        if args.save_most_recent:
            # if --save-most-recent flag is set, look for latest at a fixed filename
            resume_from = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
            if not os.path.exists(resume_from):
                # If no latest checkpoint has been saved yet, don't try to resume
                resume_from = None
        else:
            # otherwise, list checkpoint dir contents and pick the newest checkpoint
            resume_from = get_latest_checkpoint(args.checkpoint_path, remote=args.remote_sync is not None)
        if resume_from:
            logging.info(f'Found latest resume checkpoint at {resume_from}.')
        else:
            logging.info(f'No latest resume checkpoint found in {args.checkpoint_path}.')
        args.resume = resume_from

    if args.precision == 'fp16':
        logging.warning("It's recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for train.")

    # print(f'Running with a single process on device {args.device}.')

    dist_model = None

    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    log_base_path = os.path.join(args.logs, args.name)

    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model, args.pretrained, precision=args.precision, device=args.device, output_dict=True,
        aug_cfg = args.aug_cfg, )

    model = create_custom_model(args, model)  # use custom model
    
    random_seed(args.seed, args.rank)

    if args.train_data: # Keep track of the parameters of the model using params.txt
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                f.write(f"{name}: {val}\n")

    # create optimizer and scaler
    optimizer, scaler = get_optimizer_scaler(args, model)

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
            # print(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), iter=0, tokenizer=get_tokenizer(args.model), model = model)
    assert len(data), 'At least one train or eval dataset must be specified.'
    
    # create scheduler if train
    scheduler = None
    if args.train_data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        scheduler_args = {'optimizer':optimizer, 'base_lr': args.lr, 'warmup_length': args.warmup, 'steps': total_steps}
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(**scheduler_args)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(**scheduler_args)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(**(scheduler_args + [cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end]))
        else:
            logging.error(f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.train_data and args.logs and args.logs.lower() != 'none'

    writer = None
    if not args.train_data:
        classif_split = args.imagenet_val if args.imagenet_val else args.imagenet_test
        retrieval_split = args.val_data if args.val_data else args.test_data
        metrics = evaluate(model, data, start_epoch, args, writer)
        # The checkpoint contains the total epochs the model trained for, replace it with the checkpoint we want to evaluate
        # (Based on the epochs of that checkpoint)
        total_epochs = re.search("-epochs_[0-9]+", args.name)[0] 
        model_name = args.name.replace(total_epochs, f"-epochs_{start_epoch}")
        eval_file = args.eval_file if (args.val_data or args.imagenet_val) else 'test_' + args.eval_file
        with open(eval_file, 'a') as f:
            for k, v in metrics.items():
                if k == "zeroshot-val-top1":
                    f.write('{}\t{}\t{}\t{:.2f}\n'.format(model_name, classif_split, k, 100 * v))
                elif k in ["image_to_text_R@1", "image_to_text_R@5", "image_to_text_R@10",
                           "text_to_image_R@1", "text_to_image_R@5", "text_to_image_R@10"]:
                    f.write('{}\t{}\t{}\t{:.2f}\n'.format(model_name, retrieval_split, k, 100 * v))
            f.write('\n')
        return

    loss = create_loss(args)
    iterations = args.al_iter if args.al_iter is not None else 1
    for iteration in range(iterations):
        if args.active_learning:
            logging.info('Active Learning iteration: ' + str(iteration))
        if args.active_learning and iteration + 1 < args.al_iter:
            epochs = args.al_epochs
        else:
            epochs = args.epochs
        for epoch in tqdm(range(start_epoch, epochs)):
                
            train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
            completed_epoch = epoch + 1
    
            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                eval_path = os.path.join(args.logs, args.name, "val_performance.txt")
                evaluate(model, data, completed_epoch, args, writer, eval_path = eval_path)
                
            # Saving checkpoints.
            checkpoint_dict = {"epoch": completed_epoch, "name": args.name, "state_dict": model.state_dict(), 
                               "optimizer": optimizer.state_dict()}
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()
    
            # only save the last epoch to save server storage
            if completed_epoch % args.save_freq == 0: # args.epochs:
                torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"))

        # For AL, we get the data for the next iteration (hence, iteration+1) and reset our model        
        if args.active_learning and iteration + 1 < args.al_iter:
            # Fine-tune the ProbVLM model with the current CLIP model
            if args.probvlm: 
                print('ProbVLM tuning...')
                # Load the pre-trained ProbVLM adapter 
                ProbVLM_Net = get_default_BayesCap_for_CLIP()
                # Using coco_epochs + 10 to make ProbVLM get fine-tuned for 10 more epochs (resuming model at coco_epochs)
                train_ProbVLM(model, ProbVLM_Net, data['train'].dataloader, data['val'].dataloader, Cri = TempCombLoss(),
                              device='cuda', dtype=torch.float, init_lr=8e-5, num_epochs=args.coco_epochs+10, eval_every=100, 
                              ckpt_path=args.coco_save_ckpt, T1=1e0, T2=1e-4, resume_path = args.coco_resume_ckpt, log = False) 
                del ProbVLM_Net

            del data
            data = get_data(args,(preprocess_train, preprocess_val),iter=iteration+1,
                            tokenizer=get_tokenizer(args.model),model = model)
            assert len(data), 'At least one train or eval dataset must be specified.'
            del model
            model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model, args.pretrained, precision=args.precision, device=args.device, output_dict=True,
            aug_cfg = args.aug_cfg, )

            model = create_custom_model(args, model)  # use custom model
            optimizer, scaler = get_optimizer_scaler(args, model)
            
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
    return log_base_path

if __name__ == "__main__":
    main(sys.argv[1:])
