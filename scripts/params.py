import argparse
import ast
from tools import select_cpu_or_gpu

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)

def add_base_args(parser):
    parser.add_argument(
        "--train-data",type=str, default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument("--val-data", type=str, default=None, help="Path to file(s) with validation data",)
    parser.add_argument("--imagenet-val", type=str, default=None, help="Path to imagenet val set for conducting zero shot evaluation.",)

    # Custom test data params
    parser.add_argument("--test-data", type=str, default=None, help="Path to file(s) with test data",)
    parser.add_argument("--imagenet-test", type=str, default=None, help="Path to imagenet test set for conducting zero shot evaluation.",)

    # Standard parameters
    parser.add_argument( "--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument( "--epochs", type=int, default=32, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of steps to warmup for.")
    
    parser.add_argument("--zeroshot-frequency", type=int, default=1, help="How often to run zero shot.")
    
    parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: none)",)
    parser.add_argument("--precision", choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp", help="Floating point precision.")
    
    parser.add_argument("--model", type=str, default="RN50", help="Name of the vision backbone to use.",)
    parser.add_argument("--pretrained", default="openai", type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="If true, more information is logged.")
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    
    # Custom parsing arguments for S-CLIP
    parser.add_argument("--label-ratio", type=float, default=0.1, help="Subset ratio for paired data.",)
    parser.add_argument( "--method", type=str, default="base", help="Method for training (base, ours).",)
    parser.add_argument("--keyword-path", type=str, default=None, help="Path for keyword candidate set",)
    parser.add_argument("--name", type=str, default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    
    # Had to re-add these after removing them
    parser.add_argument("--logs", type=str, default='./checkpoint', help="Where to store logs. Use None to avoid storing logs.",)
    parser.add_argument("--rank", type=int, default=0, help="Rank??")
    parser.add_argument("--world_size", type=int, default=1, help="World size??")
    parser.add_argument("--distributed", action='store_true', default=False, help="Distributed??")
    parser.add_argument("--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps.")
    parser.add_argument("--lr-scheduler", type=str, default='cosine',
        help="LR scheduler. One of: 'cosine' (=default), 'const' (constant), 'const-cooldown' (constant w/ cooldown).")
    parser.add_argument("--horovod", default=False, action="store_true", help="Use horovod for distributed training.")
    parser.add_argument("--skip-scheduler", action="store_true", default=False, help="Whether to skip the learning rate decay.",)
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Gradient clip.")
    parser.add_argument("--log-every-n-steps", type=int, default=100, help="Log every n steps.",)
    parser.add_argument("--val-frequency", type=int, default=1, help="How often to run evaluation with val data.")
    parser.add_argument("--workers", type=int, default=1, help="Number of dataloader workers per GPU.")
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)

    # Own arguments for active learning
    parser.add_argument("--active-learning", action='store_true', default=False, help="Whether to apply (cold start) active learning.")
    parser.add_argument("--al-iter", type=int, default=1, help="How many times to apply active learning.")
    parser.add_argument("--al-epochs", type=int, default=5, help="How many epochs to train for during the AL phase.")
    parser.add_argument("--probvlm", action='store_true', default=False, help="Whether to use ProbVLM for active learning.")

    # Arguments for pseudo-labeling
    parser.add_argument("--pl-method", type=str, default='ot.image', help="Type of pseudo-labeling strategy to apply.")
    parser.add_argument("--use-vit", action='store_true', default=False, help="Whether to use ViT model to determine cosine sim. between images.")

    # Arguments for K-fold cross-validation
    parser.add_argument("--k-fold", type=int, default=-1, help="Which K-fold to use from the data splits.")
    parser.add_argument("--save-freq", type=int, default=-1, help="How often to save model checkpoints (after how many epochs).")
    parser.add_argument("--resume-epoch", type=int, default=None, help="How often to save model checkpoints (after how many epochs).")
    return parser


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser = add_base_args(parser)
    args = parser.parse_args(args)

    # Set some defaults 
    args.device = select_cpu_or_gpu() # Choose whether to run on CPU or GPU, depending on what's available
    if args.active_learning and args.al_iter > 1: 
        args.save_freq = -1 # Don't store results at intermediate points when using active learning
    else:
        args.al_iter = None
        args.al_epochs = None
        args.probvlm = False
    if args.save_freq == -1: # Default checkpoint-saving frequency means we only save checkpoints at the end
        args.save_freq = args.epochs
    # We don't specify any PL method if we are using the base CLIP model
    if args.method == 'base':
        args.pl_method = None

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    """Custom setup for convenience"""
    if args.method != "ours":
        args.keyword_path = None

    if args.train_data is None and args.name is not None:
        if args.resume_epoch is None:
            args.resume_epoch = "latest"
        args.resume = f"{args.name}/checkpoints/epoch_{args.resume_epoch}.pt"
        for model in ["RN50", "ViT-B-32", "ViT-B-16"]:
            if model in args.resume:
                args.model = model
        for seed in [0, 1, 2]:
            if "seed_{}".format(seed) in args.resume:
                args.seed = seed

    return args