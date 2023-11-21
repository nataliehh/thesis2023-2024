import os
import sys
import json
import numpy as np

from ds import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders, build_vocab

from utils import *
from networks import *
from train_probVLM import *

from probvlm_data_loader import get_data as probvlm_get_data

import itertools
sys.path.append('/vol/tensusers4/nhollain/thesis2023-2024/s_clip_scripts')
# from data_loader import get_data
from params import parse_args
from tools import select_cpu_or_gpu
from open_clip import create_model_and_transforms, get_tokenizer, create_loss

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
    --imagenet-val RSICD-CLS \
    '''

str_args = prep_str_args(str_args)
args = parse_args(str_args)

# Create model to use for loading data
model, preprocess_train, preprocess_val = create_model_and_transforms(args.model, args.pretrained, precision=args.precision, 
                                                                      device=args.device, output_dict=True, aug_cfg = args.aug_cfg, )

probvlm_data = probvlm_get_data((preprocess_train, preprocess_val), tokenizer=get_tokenizer(args.model), model = model, batch_size = 1024)

train_loader, valid_loader = probvlm_data['train'].dataloader, probvlm_data['val'].dataloader


# Initialize models
device = select_cpu_or_gpu()
CLIP_Net = load_model(device=device, model_path=None) # frozen clip model (?)
ProbVLM_Net = BayesCap_for_CLIP(inp_dim=512, out_dim=512, hid_dim=256, num_layers=3, p_drop=0.05,) # ProbVLM

# Train the model
train_ProbVLM(CLIP_Net,ProbVLM_Net, train_loader, valid_loader, Cri = TempCombLoss(), device=device, init_lr=8e-5,
              dtype=torch.cuda.FloatTensor, num_epochs=25, eval_every=2, ckpt_path='../ckpt/ProbVLM_Net', T1=1e0, T2=1e-4)
