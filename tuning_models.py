import sys
sys.path.append('./scripts')

import itertools
from collections import Counter
from main import main, format_checkpoint
from params import parse_args
import copy
import os
from tqdm import tqdm
from datetime import datetime

from tuning_tools import prep_str_args, evaluate_checkpoint, get_evaluation_history, gridsearch, create_base_str_args

split = 'val'
history_path = './results/test_eval.txt' if split == 'test' else './results/eval.txt' 
model_history = get_evaluation_history(history_path) # Get the history of evaluated models
eval_file = './results/eval.txt'

# Specify a method (from 'baseline', 'basic-al', 'probvlm', 'hard-pl', 'soft-pl', 's-clip')
# Specify a dataset ('RS.ALL', 'Fashion.ALL' or 'ILT' -> make sure you have the data!)
# Specify the eval file (NOTE: eval file is currently the VALIDATION split evaluation file)
# (cont. NOTE: so specify the path WITHOUT any test-set suffix/prefix!)
method = 'basic-al'
data = 'RS.ALL'
base_str_args = create_base_str_args(method, data, eval_file)

print('base str args:', base_str_args)

base_args = parse_args(prep_str_args(base_str_args))

# Specify a dictionary to gridsearch - if you want a fixed value for a parameter, provide it as '--param : [value]'
# (i.e. provide a list of length 1)
gridsearch_dict = {
    '--epochs' : [30], 
    '--lr' : [5e-4, 5e-5, 5e-6],
    '--batch-size' : [64, 128] if method != 'probvlm' else [32], # lower batch size is required to run probvlm in memory
    '--al-iter': [5, 10, 15], 
    '--al-epochs': [10],
    '--label-ratio': [0.05, 0.1, 0.2, 0.4, 0.8] if split == 'test' else [0.1],
}

gridsearch(base_str_args, gridsearch_dict, model_history, eval_file, split)