import sys
sys.path.append('/vol/tensusers4/nhollain/thesis2023-2024/s_clip_scripts')
import itertools
from params import parse_args
from main import main

def prep_str_args(str_args): # Code to parse the string style arguments, as shown below
    str_args = str_args.split('\n') # Split on newline
    str_args = [s.strip() for s in str_args] # Remove any whitespaces from the start and end of the strings
    # Split on the space between the parameter name and the value, e.g. '--name x' becomes ['--name', 'x']
    str_args = [s.split(' ') for s in str_args] 
    str_args = list(itertools.chain(*str_args)) # Flatten the resulting list of lists
    str_args = [s for s in str_args if len(s) > 0] # Remove arguments that are empty
    return str_args
    
def evaluate_checkpoint(checkpoint_path, epoch = None, kfold = -1, split = 'val'):
    # print('=> Resuming checkpoint {} (epoch {})'.format(checkpoint, epoch))
    checkpoint = checkpoint_path 
    if 'Fashion' in checkpoint:
        zeroshot_datasets = ["Fashion200k-SUBCLS", "Fashion200k-CLS", "FashionGen-CLS", "FashionGen-SUBCLS", "Polyvore-CLS", ]
        retrieval_datasets = ["FashionGen", "Polyvore", "Fashion200k",]
    else:
        zeroshot_datasets = ["RSICD-CLS", "UCM-CLS"] # "WHU-RS19", "RSSCN7", "AID" -> NOT WORKING bc of different data-loading workings
        if split == 'test': # Test split includes other remote sensing dataset
            zeroshot_datasets += ["WHU-RS19", "RSSCN7", "AID", "RESISC45"]
        retrieval_datasets = ["RSICD", "UCM", "Sydney"]
    
    for dataset in zeroshot_datasets:
        str_args = ['--name', checkpoint, f'--imagenet-{split}', dataset, '--resume-epoch', str(epoch), '--k-fold', str(kfold)]
        args = parse_args(str_args)
        main(args)
    
    for dataset in retrieval_datasets:
        str_args = ['--name', checkpoint, f'--{split}-data', dataset, '--resume-epoch', str(epoch), '--k-fold', str(kfold)]
        args = parse_args(str_args)
        main(args)
