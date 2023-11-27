import os
import sys
import json

import numpy as np

from utils import *
from networks import *
from train_probVLM import *
from probvlm_data_loader import get_data as probvlm_get_data

import matplotlib.pyplot as plt

import itertools
sys.path.append('/vol/tensusers4/nhollain/thesis2023-2024/s_clip_scripts')
from data_loader import get_data
from params import parse_args
from open_clip import create_model_and_transforms, get_tokenizer, create_loss
import torch

def prep_str_args(str_args): # Code to parse the string style arguments, as shown below
    str_args = str_args.split('\n') # Split on newline
    str_args = [s.strip() for s in str_args] # Remove any whitespaces from the start and end of the strings
    # Split on the space between the parameter name and the value, e.g. '--name x' becomes ['--name', 'x']
    str_args = [s.split(' ') for s in str_args] 
    str_args = list(itertools.chain(*str_args)) # Flatten the resulting list of lists
    str_args = [s for s in str_args if len(s) > 0] # Remove arguments that are empty
    return str_args

str_args = ''' 
    --train-data RS-ALL
    --val-data RS-ALL
    --imagenet-val RSICD-CLS 
    --label-ratio 1.0
    '''

args = parse_args(prep_str_args(str_args))

print('Creating CLIP model...')
model, preprocess_train, preprocess_val = create_model_and_transforms(args.model, args.pretrained, precision=args.precision, 
                                                                      device=args.device, output_dict=True, aug_cfg = args.aug_cfg, )

print('Getting COCO data...')
# Getting the probvlm data (COCO dataset)
probvlm_data = probvlm_get_data((preprocess_train, preprocess_val), tokenizer=get_tokenizer(args.model), batch_size = 128)
train_loader, valid_loader = probvlm_data['train'].dataloader, probvlm_data['val'].dataloader

print('Loading ProbVLM...')
CLIP_Net = load_model(device='cuda', model_path=None)
ProbVLM_Net = BayesCap_for_CLIP(inp_dim=512, out_dim=512, hid_dim=256, num_layers=3, p_drop=0.05,)

print('Loading checkpoint...')
optimizer = torch.optim.Adam(list(ProbVLM_Net.img_BayesCap.parameters())+list(ProbVLM_Net.txt_BayesCap.parameters()), lr=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 0)
ProbVLM_Net, _, _, epoch = load_checkpoint(ProbVLM_Net, optimizer, scheduler, resume_path = '../ckpt/ProbVLM_Net_last.pth')
print('Epochs trained for:', epoch)

print('Loading remote sensing data...')
rs_data = get_data(args, (preprocess_train, preprocess_val), iter=0, tokenizer=get_tokenizer(args.model), model = model)
rs_train_loader, rs_valid_loader = rs_data['train'].dataloader, rs_data['val'].dataloader

print('Making predictions for one batch...')
for image, caption in rs_train_loader:
    image, caption = image.to('cuda'), caption.to('cuda')
    z_I, z_T = CLIP_Net(image, caption)
    ProbVLM_z_I, ProbVLM_z_T = ProbVLM_Net(z_I, z_T)
    print('Prediction (images)')
    print(ProbVLM_z_I)
    print('Prediction (text)')
    print(ProbVLM_z_T)
    break