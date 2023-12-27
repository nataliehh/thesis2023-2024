import os
import sys
import json
import numpy as np

from ds import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders, build_vocab

from utils import *
from networks import *
from train_probVLM import *

from probvlm_data_loader import get_data as probvlm_get_data

from sklearn.model_selection import train_test_split

import itertools
sys.path.append('/vol/tensusers4/nhollain/thesis2023-2024/s_clip_scripts')
# from data_loader import get_data
from params import parse_args
from tools import select_cpu_or_gpu
from open_clip import create_model_and_transforms, get_tokenizer, create_loss

from datetime import datetime

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
    '''

str_args = prep_str_args(str_args)
args = parse_args(str_args)

# Create model to use for loading data
model, preprocess_train, preprocess_val = create_model_and_transforms(args.model, args.pretrained, precision=args.precision, 
                                                                      device=args.device, output_dict=True, aug_cfg = args.aug_cfg, )
# Load the image ids (=x) and their labels (=y)
x = np.load('../coco/img_ids.npy')
y = np.load('../coco/labels.npy')

label_ratios = sorted([1.0], reverse = True) # 0.8


for label_ratio in label_ratios:
    t_start = datetime.now()
    if label_ratio == 1:
        x_train = x
    else:
        x_train, x_test = train_test_split(x, train_size = label_ratio, stratify = y, random_state = 0)
    np.save('../coco/train_img_ids.npy', x_train)
    with open('eval.txt', 'a') as f:
        print('Label ratio:', label_ratio)
        f.write(f'label_ratio: {label_ratio}\n')

    probvlm_data = probvlm_get_data((preprocess_train, preprocess_val), tokenizer=get_tokenizer(args.model), 
                                    batch_size = 256, label_ratio = 1)
    
    train_loader, valid_loader = probvlm_data['train'].dataloader, probvlm_data['val'].dataloader
    
    
    # Initialize models
    device = 'cuda' #select_cpu_or_gpu()
    CLIP_Net = model #load_model(device=device, model_path=None) # frozen clip model (?)
    ProbVLM_Net = get_default_BayesCap_for_CLIP() # ProbVLM
    
    # Train the model
    #resume_path = '../ckpt/ProbVLM_Net_last.pth',
    train_ProbVLM(CLIP_Net,ProbVLM_Net, train_loader, valid_loader, Cri = TempCombLoss(), device=device, 
                  resume_path = f'../ckpt/ProbVLM_Net_label_ratio_{label_ratio}_last.pth',
                  dtype=torch.cuda.FloatTensor, num_epochs=100, eval_every=5, init_lr=8e-5, T1=1e0, T2=1e-4
                  ckpt_path=f'../ckpt/ProbVLM_Net_label_ratio_{label_ratio}',)
    t_delta = datetime.now() - t_start
    print(f'Elapsed time: {t_delta}')

    with open('eval.txt', 'a') as f:
        f.write(f'Elapsed time: {t_delta}\n')

