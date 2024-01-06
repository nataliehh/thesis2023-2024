import sys
sys.path.append('./scripts')
import itertools
from params import parse_args
from main import main, format_checkpoint

from collections import Counter
from params import parse_args
import copy
import os
from tqdm import tqdm
from datetime import datetime
import time


def prep_str_args(str_args): # Code to parse the string style arguments, as shown below
    str_args = str_args.split('\n') # Split on newline
    str_args = [s.strip() for s in str_args] # Remove any whitespaces from the start and end of the strings
    # Split on the space between the parameter name and the value, e.g. '--name x' becomes ['--name', 'x']
    str_args = [s.split(' ') for s in str_args] 
    str_args = list(itertools.chain(*str_args)) # Flatten the resulting list of lists
    str_args = [s for s in str_args if len(s) > 0] # Remove arguments that are empty
    return str_args

# Compute how many evaluations are done each time the model is called. 
# This is done because each model is evaluated using multiple metrics and on multiple dataset (subsets). 
# E.g. if we evaluate on [(dataset1, metric1), (dataset2, metric1), ..., (datasetn, metric1), ... (datasetn, metricn)]. 
# Then the length of this list is how often the model is evaluated when it is called.
def get_num_evals(zeroshot_datasets, retrieval_datasets):
    zeroshot_num = 1 # We compute classification accuracy once
    retrieval_num = 6 # We compute recall @ 1, 5 and 10, for both text-to-image and image-to-text (=3*2)
    total_evals = zeroshot_num * len(zeroshot_datasets) + retrieval_num * len(retrieval_datasets)
    return total_evals

# Obtain the datasets for which we evaluate, based on the type of data we're working with
def get_eval_datasets(checkpoint, split, dataset):
    possible_datasets = ['Fashion', 'ILT', 'RS']
    if 'Fashion' in dataset or 'Fashion' in str(checkpoint):
        zeroshot_datasets = ["Fashion200k-SUBCLS", "Fashion200k-CLS", "FashionGen-CLS", "FashionGen-SUBCLS", "Polyvore-CLS", ]
        retrieval_datasets = ["FashionGen", "Polyvore", "Fashion200k",]
    elif 'ILT' in dataset or 'ILT' in str(checkpoint):
        zeroshot_datasets = ['ILT-CLS']
        retrieval_datasets = ['ILT']
    elif 'RS' in dataset or 'RS' in str(checkpoint):
        zeroshot_datasets = ["RSICD-CLS", "UCM-CLS"] 
        if split == 'test': # Test split includes other remote sensing dataset
            zeroshot_datasets += ["WHU-RS19", "RSSCN7", "AID", "RESISC45"]
        retrieval_datasets = ["RSICD", "UCM", "Sydney"] 
    else:
        print('No known dataset for this checkpoint. Possible datasets are', possible_datasets)
        zeroshot_datasets, retrieval_datasets = None, None
    return zeroshot_datasets, retrieval_datasets

def evaluate_checkpoint(checkpoint_path, epoch = 0, kfold = -1, split = 'val', dataset = 'RS', eval_file = ''):
    # print('=> Resuming checkpoint {} (epoch {})'.format(checkpoint, epoch))
    checkpoint = checkpoint_path
    # Get the datasets to evaluated on
    zeroshot_datasets, retrieval_datasets = get_eval_datasets(checkpoint, split, dataset)
    
    for dataset in zeroshot_datasets:
        lst_args = [f'--imagenet-{split}', dataset, '--resume-epoch', str(epoch), '--k-fold', str(kfold)]
        if checkpoint is not None:
            lst_args += ['--name', checkpoint]
        if len(eval_file) > 0:
            lst_args += ['--eval-file', eval_file]
        args = parse_args(lst_args)
        main(args)
    
    for dataset in retrieval_datasets:
        lst_args = [f'--{split}-data', dataset, '--resume-epoch', str(epoch), '--k-fold', str(kfold)]
        if checkpoint is not None:
            lst_args += ['--name', checkpoint]
        if len(eval_file) > 0:
            lst_args += ['--eval-file', eval_file]
        args = parse_args(lst_args)
        main(args)
        
def get_evaluation_history(eval_file):
    # Count the models that have been trained and evaluated already, based on their parameters 
    results = []
    if os.path.exists(eval_file):   
        with open(eval_file, 'r') as f:
            results = f.readlines()

    # Remove any non-result lines from the eval file, and split the lines on the tab character
    # (results have format: model_name\tdataset_name\tmetric_name\tmetric_value)
    results = [r.replace('\n','').split('\t')[0] for r in results if '\t' in r]
    model_history = results
    # Remove the timestamp from the model names, as well as the specific fold - rest of the name contains params
    model_history = ['-'.join(m.split('-')[2:]).split('-fold')[0] for m in model_history]
    model_history = dict(Counter(model_history))
    return model_history

# Returns the default base arguments for the models. NOTE: parameters that are tuned will NOT be added to args here!!!
def create_base_str_args(method:str, data:str, eval_file:str, keyword_path:str = '', pl_modality = ''):
    
    base_str_args = f''' 
    --train-data {data}
    --val-data {data}
    --save-freq 5
    --eval-file {eval_file}
    --zeroshot-frequency 5
    '''
    base_methods = ['baseline']
    al_methods = ['basic-al', 'probvlm']
    pl_methods = ['hard-pl', 'soft-pl', 's-clip'] 
    
    methods = base_methods + al_methods + pl_methods
    if method not in methods:
        print('Unknown method. Possible methods are', methods)
        return None
    if method in al_methods:
        base_str_args += '\n--active-learning'
        if method =='probvlm':
            base_str_args += '\n--probvlm'
            base_str_args += '\n--device cuda'
    if method in pl_methods:
        base_str_args += '\n--method ours'
        if len(keyword_path) == 0: # Check if keyword path is specified, warn if it's not
            print('Did not specify keyword_path, will use default path based on the chosen dataset')
            keyword_path = f'./keywords/{data}/class-name.txt'
        if os.path.exists(keyword_path):
            print('Using default keyword path:', keyword_path)
        else:
            print('WARNING: keyword path is not an existing path. Example of a correct path: "keywords/RS/class-name.txt"') 
        base_str_args += f'\n--keyword-path {keyword_path}'
        if len(pl_modality) > 0:
            # the pl-method argument looks like hard.image, ot.image, soft.image, hard.text, etc.
            # here, we format it if the modality has been provided
            pl_name= 'ot' if method == 's-clip' else 'hard' if method == 'hard-pl' else 'soft'
            base_str_args += f'\n--pl-method {pl_name}.{pl_modality}'
    else:
        base_str_args += '\n--method base'
    return base_str_args

# Gridsearch params: string of base arguments, dict of gridsearch arguments, how often to repeat the training + eval of a model config
# And the number of evaluations that are done per model (to be able to calculate how often a config has been used already)
def gridsearch(base_str_args:str, gridsearch_dict:dict, model_history:dict, eval_file:str, split:str, num_repeats:int = 5, change_seed:bool = False):
    '''
    base_str_args (str): a string with the base arguments for all models in the gridsearch. 
                    Example of format: "--train-data RSICD\n--val-data RSICD"
    gridsearch_dict (dict): a dictionary with the parameters to search.
                    Example of format: {"--lr" : [5e-4, 5e-5, 5e-6], "--batch-size" : [32, 64, 128]}
    model_history (dict): a dictionary which contains the number of evaluations a model configuration has had. Can be obtained with the function "get_evaluation_history" (with the same eval_file as here).
                    Example of format: {"model-name-1": 20, "model-name-2": 40}
    eval_file (str): the path to the evaluation file where we want to store the results of evaluating the model configs.
                    Example of format: "./results/eval.txt"
    split (str): which split to evaluate on, should be "val" for the validation split or "test" for the test split of a dataset.
    num_repeats (int): the number of times we evaluate each model configuration in the gridsearch. Default: 5. 
    change_seed (bool): Whether to change the default seed (=42). If True, which repeat we are on for the config determines the seed.
    '''
    zeroshot_datasets, retrieval_datasets = get_eval_datasets(base_str_args, split, dataset = '')
    num_evals_per_run = get_num_evals(zeroshot_datasets, retrieval_datasets)
    print('Evaluations per model run:', num_evals_per_run)

#     print('Model history', model_history)
    # Get a list of the gridsearch parameters
    gridsearch_values = list(gridsearch_dict.values())
    gridsearch_keys = list(gridsearch_dict.keys())
    configs = list(itertools.product(*gridsearch_values))
    print('Number of configs:', len(configs))
    
    t_start = datetime.now()  # Keep track of time
    
    # Loop over the configs -> gridsearch
    for c, config in enumerate(configs): 
        str_args = copy.deepcopy(base_str_args)
        # Add the gridsearch parameters to the base arguments
        for i, param in enumerate(config):
            param_name = gridsearch_keys[i]
            str_args += '\n{} {}'.format(param_name, param)
        print('str args:', str_args)

        str_args = prep_str_args(str_args)
        print(str_args)
        args = parse_args(str_args)
        checkpoint_hypothetical = format_checkpoint(args)
        # Remove the timestamp from the hypothetical checkpoint, so we can compare to the params of other checkpoints
        checkpoint_params = '-'.join(checkpoint_hypothetical.split('-')[2:]).split('-fold')[0]

        # Check if we've already trained the exact same model, correct the number of training iterations we still need to do
        if checkpoint_params in model_history:
            # The number of times to repeat depends on how often the model's been evaluated already
            num_evals = model_history[checkpoint_params]
            print('num_evals already done of this model config:', num_evals)
            start_repeat = int(num_evals/num_evals_per_run)
        else: # If we've never trained + evaluated the model before, just use num_repeats
            start_repeat = 0
        print(f'Config number {c}: {max(0,num_repeats-start_repeat)} repeats')
        for i in range(start_repeat, num_repeats):
            args = parse_args(str_args)
            if change_seed: # Set seed based on the number of repeats
                args.seed = i
            # We compute here for which epochs we need to evaluate (based on for which epochs we checkpoint)
            epoch_freq = args.save_freq
            epochs = list(range(epoch_freq,args.epochs+1,epoch_freq))

            checkpoint_path = main(args) # Calls the main.py function of S-CLIP
            for epoch in epochs:
                t_1 = time.time()
                evaluate_checkpoint(checkpoint_path, epoch = epoch, split = split, eval_file = eval_file)
                print(f'Evaluation time: {round(time.time()-t_1, 3)}')
            # Remove the checkpoint after evaluating, to save space
            os.system(f"rm -r {checkpoint_path}")  

    t_delta = datetime.now() - t_start   
    print(f'Elapsed time: {t_delta}')
