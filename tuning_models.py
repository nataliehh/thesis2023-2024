import sys
sys.path.append('/vol/tensusers4/nhollain/thesis2023-2024/s_clip_scripts')

import itertools
from collections import Counter
from main import main, format_checkpoint
from params import parse_args
import copy
import os
from tqdm import tqdm
from datetime import datetime

def prep_str_args(str_args): # Code to parse the string style arguments, as shown below
    str_args = str_args.split('\n') # Split on newline
    str_args = [s.strip() for s in str_args] # Remove any whitespaces from the start and end of the strings
    # Split on the space between the parameter name and the value, e.g. '--name x' becomes ['--name', 'x']
    str_args = [s.split(' ') for s in str_args] 
    str_args = list(itertools.chain(*str_args)) # Flatten the resulting list of lists
    str_args = [s for s in str_args if len(s) > 0] # Remove arguments that are empty
    return str_args
    
def evaluate(checkpoint, epoch = None, kfold = -1):
    # print('=> Resuming checkpoint {} (epoch {})'.format(checkpoint, epoch))
    checkpoint = checkpoint_path 
    if 'Fashion' in checkpoint:
        zeroshot_datasets = ["Fashion200k-SUBCLS", "Fashion200k-CLS", "FashionGen-CLS", "FashionGen-SUBCLS", "Polyvore-CLS", ]
        retrieval_datasets = ["FashionGen", "Polyvore", "Fashion200k",]
    else:
        zeroshot_datasets = ["RSICD-CLS", "UCM-CLS"] # "WHU-RS19", "RSSCN7", "AID" -> NOT WORKING bc of different data-loading workings
        retrieval_datasets = ["RSICD", "UCM", "Sydney"]
    
    for dataset in zeroshot_datasets:
        str_args = ['--name', checkpoint, '--imagenet-val', dataset, '--resume-epoch', str(epoch), '--k-fold', str(kfold)]
        args = parse_args(str_args)
        main(args)
    
    for dataset in retrieval_datasets:
        str_args = ['--name', checkpoint, '--val-data', dataset, '--resume-epoch', str(epoch), '--k-fold', str(kfold)]
        args = parse_args(str_args)
        main(args)

########################
# Count the models that have been trained and evaluated already, based on their parameters 
results = []
if os.path.exists('./eval.txt'):   
    with open('./eval.txt', 'r') as f:
        results = f.readlines()

# Remove any non-result lines from the eval file, and split the lines on the tab character
# (results have format: model_name\tdataset_name\tmetric_name\tmetric_value)
results = [r.replace('\n','').split('\t')[0] for r in results if '\t' in r]
model_names = results
# Remove the timestamp from the model names, as well as the specific fold - rest of the name contains params
model_names = ['-'.join(m.split('-')[2:]).split('-fold')[0] for m in model_names]
model_names = dict(Counter(model_names))

# Do a grid search on the parameters
# NOTE: for active learning, save-freq should be set to 1
base_str_args = ''' --train-data RS-ALL
--val-data RS-ALL
--imagenet-val RSICD-CLS 
--keyword-path keywords/RS/class-name.txt
--zeroshot-frequency 5  
--method ours
--save-freq 5
'''
# --label-ratio 0.1
# --active-learning

# Dictionary of values to gridsearch for hyperparam tuning
gridsearch_dict = {
    '--epochs' : [35], #list(range(15,36,5)) if 'active-learning' in base_str_args else [35], #[10,15,20,25,30,35],
    '--lr' : [5e-4, 5e-5, 5e-6], # 5e-4, 5e-6
    '--batch-size' : [128],
    '--al-iter': [1], #list(range(3,17,2)), #list(range(1,6,2)),
    '--al-epochs': [35],
    '--label-ratio': [0.1],
    '--pl-method': ["soft.image", "soft.text", "ot.image", "ot.text", "hard.image", "hard.text"],
}

# This number is very specifically chosen because we have 9 folds for the datasets!
num_repeats = 9
num_evals = 20 # How many evaluations are done with evaluate(...) - KEEP THIS FIXED

gridsearch_values = list(gridsearch_dict.values())
gridsearch_keys = list(gridsearch_dict.keys())
configs = list(itertools.product(*gridsearch_values))
print('Number of configs:', len(configs))

t_start = datetime.now() 
for c, config in enumerate(configs): # Gridsearch
    str_args = copy.deepcopy(base_str_args)
    # Add the gridsearch parameters to the arguments
    for i, param in enumerate(config):
        param_name = gridsearch_keys[i]
        str_args += '\n{} {}'.format(param_name, param)
        
    str_args = prep_str_args(str_args)
    print(str_args)
    args = parse_args(str_args)
    checkpoint_hypothetical = format_checkpoint(args)
    # Remove the timestamp from the hypothetical checkpoint, so we can compare to the params of other checkpoints
    checkpoint_params = '-'.join(checkpoint_hypothetical.split('-')[2:]).split('-fold')[0]

    # Check if we've already trained the exact same model, correct the number of training iterations we still need to do
    if checkpoint_params in model_names:
        # The number of times to repeat depends on how often the model's been evaluated already
        start_repeat = int(model_names[checkpoint_params]/num_evals)
    else: # If we've never trained + evaluated the model before, just use num_repeats
        start_repeat = 0
    print(f'Config number {c}: {max(0,num_repeats-start_repeat)} repeats')
    for i in range(start_repeat, num_repeats):
        # print('repeat number', i) 
        args = parse_args(str_args)
        args.k_fold = i
        # We compute here for which epochs we need to evaluate (based on for which epochs we checkpoint)
        epoch_freq = args.save_freq
        epochs = list(range(epoch_freq,args.epochs+1,epoch_freq))
        # print('Epochs to checkpoint', epochs)
        # print('Args k fold (outside):' , args.k_fold)
        checkpoint_path = main(args) # Calls the main.py function of S-CLIP
        for epoch in epochs:
            evaluate(checkpoint_path, epoch = epoch, kfold = i)
        # Remove the checkpoint after evaluating, to save space
        os.system(f"rm -r {checkpoint_path}")  

t_delta = datetime.now() - t_start   
print(f'Elapsed time: {t_delta}')
