# Code based on: https://github.com/alinlab/s-clip/blob/master/custom/data.py
# And on: https://github.com/isaaccorley/torchrs/blob/main/torchrs/datasets/
import os
import h5py
import random
from PIL import Image
from multiprocessing import Value
from typing import List, Dict
from tqdm import tqdm
import time
import numpy as np
import gc


import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

from datasets import load_dataset

import json

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


class DataInfo:
    def __init__(self, dataloader: DataLoader, sampler: DistributedSampler = None, shared_epoch: SharedEpoch = None):
        self.dataloader = dataloader
        self.sampler = sampler
        self.shared_epoch = shared_epoch

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class Coco(torch.utils.data.Dataset):
    def __init__(self, root: str , split: str = 'train', transform: T.Compose = None, image_root: str = '', ):
        self.root = root 
        self.split = split
        self.transform = transform
        data_anns = "captions_{}2014.json".format(self.split)
        self.train_ids = np.load(os.path.join(self.root,"train_img_ids.npy"))
        self.data = self.load_captions(os.path.join(self.root, data_anns))
        self.image_root = "images"
        
    def load_captions(self, path: str):
        with open(path, 'r') as f:
            # contains img_id, caption_id and captions
            content = json.load(f)['annotations']
        captions = {}
        for annotation in content:
            image_id, _, caption = annotation.values() 
            # Skip train instances if they are not within our selection of image_ids
            if self.split == "train" and image_id not in self.train_ids:
                continue
            caption = caption.replace('\n', '.') # Some captions end in a newline, instead of a period
            if image_id in captions:
                captions[image_id].append(caption)
            else:
                captions[image_id] = [caption]
        return list(captions.items())
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        img_id, caption_lst = self.data[idx]
        # example path: coco/images/val2014/COCO_val2014_000000442206.jpg
        # All image IDs are padded to a length of 12 with zero-padding (e.g. 000000442206)
        img_path = os.path.join(self.root, self.image_root) + '/{0}2014/COCO_{0}2014_{1}.jpg'
        img_path = img_path.format(self.split, str(img_id).zfill(12))
        x = Image.open(img_path).convert('RGB')
        x = self.transform(x)
        return dict(x=x, captions=caption_lst)
        

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_key=None, text_key=None,
                 tokenizer=None, keywords=None):
        self.dataset = dataset
        self.image_key = image_key
        self.text_key = text_key
        self.tokenize = tokenizer or (lambda x: x)

        self.keywords = keywords
        self.keyword_tokens = self._init_keyword_tokens()

    def _init_keyword_tokens(self):
        if self.keywords is not None:
            BOS, EOS = 49406, 49407
            keyword_tokens = []
            for k in self.keywords:
                k = self.tokenize(k).flatten().tolist()
                k = k[k.index(BOS) + 1: k.index(EOS)] # Remove BOS, EOS from tokens
                keyword_tokens.append(k)
            return keyword_tokens
        else:
            return None

    def _find_keyword(self, tokens, key):
        for i in range(len(tokens)):
            idx = i  # candidate
            for j in range(len(key)):
                if tokens[i+j] != key[j]:
                    idx = None
                    break

            if idx is not None:
                return idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[int(idx)]

        # read data, which is dict or list
        if isinstance(data, (list, tuple)):
            images, texts = data
        else:
            assert isinstance(data, dict)
            assert self.image_key and self.text_key
            images = data[self.image_key]
            texts = data[self.text_key]

        # tokenize captions
        if isinstance(texts, list):
            texts = random.choice(texts)
        if isinstance(texts, dict):
            texts = texts['raw']
        tokens = self.tokenize([texts])[0]
        # done if not using keywords
        if self.keywords is None:
            return images, tokens

        # logics for parsing keyword labels
        keyword_labels = torch.zeros(len(self.keywords), 3)
        spaced = lambda word: " {} ".format(word)
        for i, k in enumerate(self.keywords):
            if spaced(k) in spaced(texts):
                # find index of the keyword
                key = self.keyword_tokens[i]
                idx = self._find_keyword(tokens.tolist(), key)

                assert all(tokens[idx+i] == key[i] for i in range(len(key)))

                keyword_labels[i][0] = 1
                keyword_labels[i][1] = idx
                keyword_labels[i][2] = len(key)

        return images, tokens, keyword_labels


def read_keywords(path):
    keywords = []
    with open(path, "r") as f:
        for line in f.readlines():
            keywords.append(line.strip())
    return keywords


def create_datainfo(dataset, batch_size, is_train):
    num_samples = len(dataset)
    sampler = None
    shuffle = is_train and sampler is None
    workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def split_data(d, split_ratio, seed=42, hf_data=False):
    # set random seed
    gen = torch.Generator()
    gen.manual_seed(seed)

    # split labeled and unlabeled data
    indices = torch.randperm(len(d), generator=gen)
    size = int(len(d) * split_ratio)

    if hf_data is False:
        d1 = Subset(d, indices[:size])
        d2 = Subset(d, indices[size:])
    else:
        d1 = [d[int(i)] for i in indices[:size]]
        d2 = [d[int(i)] for i in indices[size:]]

    return d1, d2

# Coco dataset: https://cocodataset.org/#download
def get_custom_data(preprocess_fn, is_train, **data_kwargs):
    path = '/vol/tensusers4/nhollain/ProbVLM/'
    split = "train" if is_train else "val"
    print('Coco data (split: {})'.format(split), end = '\t')

    # Load Coco data
    d = Coco(os.path.join(path, "coco"), split = split, transform=preprocess_fn)

    # We tokenize the caption data
    d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)
    return d


def get_data(preprocess_fns, tokenizer=None, train_data = True, val_data = True, batch_size = 64, label_ratio = 0.1):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if val_data:
        d_val = get_custom_data(preprocess_fn = preprocess_val, is_train=False, tokenizer=tokenizer)
        data["val"] = create_datainfo(d_val, batch_size, is_train=False)

    if train_data:
        d_train = get_custom_data(is_train = True, preprocess_fn = preprocess_train, tokenizer = tokenizer)
        d_train, _ = split_data(d_train, label_ratio)
        data["train"] = create_datainfo(d_train, batch_size, is_train=True)

    return data