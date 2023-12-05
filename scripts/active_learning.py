import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.functional import pairwise_cosine_similarity
from precision import get_autocast
from open_clip import get_cast_dtype, get_tokenizer

import sys
sys.path.append('/vol/tensusers4/nhollain/ProbVLM/src') # Allow probvlm imports

from utils import get_features_uncer_ProbVLM, sort_wrt_uncer
from networks import BayesCap_for_CLIP

def basic_AL(args, data, model, classnames = None, templates = None):
   # Keep track of the image & text features
    embed_size = 1024 # This is the embedding size of the CLIP model
    image_features = torch.empty((0,embed_size)).to(args.device)
    text_features = torch.empty((0,embed_size)).to(args.device)
    
    # Set some params to make inference with CLIP fit into memory
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with autocast():
        tokenizer = get_tokenizer(args.model)
        for classname in classnames: 
            texts = [template(classname) for template in templates]
            texts = tokenizer(texts).to(args.device, non_blocking=True)
            text_feature = model.encode_text(texts)
            text_features = torch.cat((text_features, text_feature), 0)

    for images, _ in data:
        images = images.to(args.device, non_blocking=True)
        if cast_dtype is not None:
            images = images.to(dtype=cast_dtype, non_blocking=True)
        with autocast():
            image_feature = model.encode_image(images)

        image_features = torch.cat((image_features, image_feature), 0)

    # Compute similarity
    sim = pairwise_cosine_similarity(image_features, text_features) #1 - cdist(image_features, text_features, metric = 'cosine')
    sim = torch.nn.functional.softmax(sim, -1) # We use softmax to get prediction probabilities

    # Compute the entropy for each similarity: sim*log(sim), we add a small constant to avoid log(0)
    entropy = torch.mul(sim,torch.log(sim+0.000001)) 
     
    # The uncertainty is the (negative) sum per row of the entropy
    uncertainty = torch.sum(-1*entropy, 1) 
    idx_by_uncertainty = uncertainty.argsort(descending = True) # Provides the indices in order of uncertainty (from high to low)
    return idx_by_uncertainty

def probvlm_AL(data, model, resume_path = ''):
    CLIP_Net = model 
    # Load the pre-trained ProbVLM adapter 
    ProbVLM_Net = BayesCap_for_CLIP(inp_dim=512, out_dim=512, hid_dim=256, num_layers=3, p_drop=0.05,)
    checkpoint = torch.load(resume_path)
    ProbVLM_Net.load_state_dict(checkpoint['model'])

    # Get a dictionary of the probabilistic parameters 
    r_dict = get_features_uncer_ProbVLM(CLIP_Net, ProbVLM_Net, data)
    
    # Get the (sorted) uncertainties for the images (discard text uncertainty)
    uncertainty_images, _ = sort_wrt_uncer(r_dict)
    # The uncertainties are given as (idx, uncertainty), we extract only the idx
    idx_by_uncertainty = [u[0] for u in uncertainty_images]
    return idx_by_uncertainty