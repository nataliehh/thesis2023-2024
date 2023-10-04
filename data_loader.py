# Code based on: https://github.com/alinlab/s-clip/blob/master/custom/data.py
# And on: https://github.com/isaaccorley/torchrs/blob/main/torchrs/datasets/
import os
import json
import h5py
import random
from PIL import Image
from multiprocessing import Value
from typing import List, Dict

from tools import read_json

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from datasets import load_dataset

from torchrs.datasets import UCMCaptions, SydneyCaptions
from torchrs.datasets import UCM, WHURS19, RSSCN7, AID, RESISC45

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class CustomDataLoader(torch.utils.data.Dataset):
    
    splits = ["train", "val", "test"]
    def __init__(self, root: str , split: str = 'train', transform: T.Compose = None, image_root: str = '', rawsent: bool = False, 
                 randomitem: bool = False, subclass: bool = False, cls: bool = False):
        ''' Initialize data loader.
        root: path to the dataset folder with images + labels.
        split: which split to use (train, validation or test set).
        transform: type of image transformation to perform.
        image_root: the folder within root where the images are contained.
        rawsent: whether we should obtain the 'raw' sentences from item['sentences'].
        randomitem: whether we should pick a random item/image in get_item.
        subclass: whether to use the subclasses as opposed to the main classes of the dataset.
        cls: True if we want to use classification labels for y instead of captions.
        '''
        self.root = root 
        self.split = split
        self.transform = transform
        self.image_root = image_root
        self.rawsent = rawsent
        self.randomitem = randomitem
        self.subclass = subclass
        self.cls = cls
        self.data = [] # Initialize as empty in parent class
        
    def __len__(self) -> int:
        return len(self.data)
    
    # Get the x value (= image) of an item, and transform it
    def get_x(self, item):
        if self.randomitem:
            path = os.path.join(self.root, self.image_root, random.choice(item["image_path"]))
        else:
            path = os.path.join(self.root, self.image_root, item["image_path"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        return x
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        x = self.get_x(item) # Get the x value (= image)
        # If we want to do classification, we use the class names as labels for the images
        if self.cls:
            cls_name = item['class_name']
            y = self.classes.index(cls_name)
            return x, y
        sentences = item["sentences"] # y labels are the sentences of an image x
        # For some datasets, we want to only extract the 'raw' sentences (and not also the tokenized versions)
        if self.rawsent: 
            sentences = [sentence["raw"] for sentence in sentences]
        
        return dict(x=x, captions=sentences)

# Data from: https://github.com/201528014227051/RSICD_optimal
class RSICD(CustomDataLoader):
    """ Data from: https://arxiv.org/abs/1712.07835 """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.load_captions(os.path.join(self.root, "dataset_rsicd.json"), self.split)
        self.image_root = "RSICD_images"
        if self.cls:
            self.load_class_info(os.path.join(self.root, "txtclasses_rsicd"))

    @staticmethod
    def load_captions(path: str, split: str) -> List[Dict]:
        captions = read_json(path)["images"]
        return [c for c in captions if c["split"] == split]
    
    def load_class_info(self, class_dir):
        classes = []
        path2class = {}
        for idx, fn in enumerate(sorted(os.listdir(class_dir))):
            classes.append(fn.split(".txt")[0])
            with open(os.path.join(class_dir, fn)) as f:
                for line in f.readlines():
                    path2class[line.strip()] = idx

        self.classes = classes
        self.path2class = path2class
    
    def __getitem__(self, idx):
        if self.cls:
            item = self.data[idx]
            x = self.get_x(item)
            y = self.path2class[item['image_path']]
            return x, y
        else:
            return super().__getitem__(idx)

# Data from: https://github.com/xthan/fashion-200k/tree/master
class Fashion200k(CustomDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self._load_annotation_db(self.split)

        if self.cls:
            # Remove some broken links
            self.data = [item for item in self.data if os.path.exists(os.path.join(self.root, 'women', item["image_path"]))]

            # Get a list of non-duplicate class names, and sort them
            self.classes = set([item['class_name'] for item in self.data])
            self.classes = sorted(self.classes)

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'test'}[split]

        txt_path = ['dress', 'jacket', 'pants', 'skirt', 'top']
        txt_format = 'labels/{}_{}_detect_all.txt' # Need to provide the fashion class and dataset split

        if self.cls:
            # We concat all label files in the split to allow us to easily get all image paths and classes
            path = './data/fashion200k/labels'
            full_txt = []
            # Go over all text files
            for txt in os.listdir(path):
                if split in txt: # If the text file contains the correct split ('train' or 'test')
                    with open(os.path.join(path, txt), 'r') as f:
                        data = f.readlines()
                        full_txt += data
            
            class_index = 2 if self.subclass else 1
            split = {'train': 'train', 'val': 'test'}[split]
            # json_path = os.path.join(self.root, f"{split}_info.json")
            # anno_json = read_json(json_path)

            data = []
            for item in full_txt:
                # for image_path in item['images']:
                # The lines in the labels have the format: path\tconfidence_score\tdescription
                image_path = item.split('\t')[0] # We extract the path to the image
                # The path has the format: /women/category/subcategory/id_folder/image, we want either the category or subcategory
                class_name = image_path.split("/")[class_index].replace("_", " ") 
                data.append({ "image_path": image_path, "class_name": class_name })
        else:
            data = {}
            for txt in txt_path:
                with open(os.path.join(self.root, txt_format.format(txt, split)), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        image_path, _, sentences = line.split('\t')
                        item_id = image_path.split('/')[3]

                        if not os.path.exists(os.path.join(self.root, image_path)):
                            continue

                        if item_id in data:
                            data[item_id]['image_path'].append(image_path)
                        else:
                            data[item_id] = dict(image_path=[image_path], sentences=sentences)
            data = [dict({'id': item_id}, **data[item_id]) for item_id in data]
        return data

# Data from: https://github.com/xthan/polyvore-dataset/tree/master
class Polyvore(CustomDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self._load_annotation_db(self.split)
        self.image_root = 'images'

        if self.cls:
            # Get a list of non-duplicate class names, and sort them
            self.classes = set([item['class_name'] for item in self.data])
            self.classes = sorted(self.classes)

    def _load_annotation_db(self, split):
        json_path = os.path.join(self.root, f"{split}_no_dup.json")
        
        anno_json = read_json(json_path)
        
        data = []

        if self.cls:
            metadata_path = os.path.join(self.root, "polyvore_item_metadata.json")
            meta_json = read_json(metadata_path)

            for item in anno_json:
                # NOT WORKING!!!
                data.append({"image_path": item["images"], "class_name": meta_json[str(item['id'])]['semantic_category']})
        else:
            for item in anno_json:
                # NOT WORKING!!!
                data.append({"image_path": item["images"], "sentences": item["title"] + "." + item["description"]})

        return data

# Data from: https://github.com/tingyaohsu/SciCap
class SciCap(CustomDataLoader):
    MAXLEN = 77  # maximum length for caption
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_root = os.path.join("SciCap-No-Subfig-Img", self.split)
        self.data = self._init_data()

    def _init_data(self):
        json_root = os.path.join(self.root, "SciCap-Caption-All", self.split)

        data = []
        for filename in os.listdir(json_root):
            json_object = read_json(os.path.join(json_root, filename))
            if json_object["contains-subfigure"]: # Only keep images without subfigures for simplicity
                continue

            path = str(filename).replace("json", "png")
            caption = json_object['0-originally-extracted']
            caption = caption[:self.MAXLEN]  # cut long captions
            data.append({'image_path': path, 'class_name': caption})

        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        x = self.get_x(item)
        return x, item['class_name']


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
                k = k[k.index(BOS) + 1: k.index(EOS)]
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
            texts = str(random.choice(texts))
        tokens = self.tokenize([str(texts)])[0]

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


def read_keywords(path):
    keywords = []
    with open(path, "r") as f:
        for line in f.readlines():
            keywords.append(line.strip())
    return keywords


def create_datainfo(args, dataset, batch_size, is_train):
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    workers = args.workers if not args.train_data else 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

# The links to most datasets were listed above, other datasets may be available via these scripts: https://github.com/isaaccorley/torchrs/tree/main/scripts
def get_custom_data(args, data, preprocess_fn, is_train, cls, subclass, **data_kwargs):
    path = './data/'
    split = "train" if is_train else "val"

    config = { # config dictionary that contains the call function to the dataset creator and the (relative) path to its data
        "RSICD": (RSICD, "RSICD", True),
        "UCM": (UCMCaptions, "UCM", False),
        "Sydney": (SydneyCaptions, "sydney_captions", False),
        "UCM-CLS": (UCM, "UCMerced_LandUse", False),
        "WHU-RS19": (WHURS19, "WHU-RS19", False),
        "RSSCN7": (RSSCN7, "RSSCN7", False),
        "AID": (AID, "AID", False),
        "RESISC45": (RESISC45, "NWPU-RESISC45", False),
        "Fashion200k": (Fashion200k, "fashion200k", True),
        "FashionGen": (None, None, None, None),  # Currently unavailable
        "Polyvore": (Polyvore, "polyvore", True),
        # "Simpsons-Captions": (None, "simpsons-blip-captions", False),
        # "Simpsons-Images": (None, "simpsons_dataset", False)
    }
    REMOTE_SENSING = ["RSICD", "UCM", "Sydney", "RS-ALL", "WHU-RS19", "RSSCN7", "AID", "RESISC45"]

    # We use a dictionary for the configuration of each dataset loader
    if data in config:
        # A configuration specifies the class instantiation, path to the dataset and whether we are using a custom dataset loader
        # Custom dataset loaders allow for extra arguments
        dataset_class, dataset_path, custom = config.get(data) 
        randomitem = True if data == 'Fashion200k' else False
        if custom:
            d = dataset_class(os.path.join(path, dataset_path), split = split, transform=preprocess_fn, cls = cls, subclass = subclass, randomitem = randomitem)
        else:
            d = dataset_class(os.path.join(path, dataset_path), transform=preprocess_fn)
        if cls:
            # Classification datasets use a captioning template, either 'a photo of [CLASS]' or 'an aerial photograph of [CLASS]' (for remote sensing)
            template = [lambda c: f"a photo of a {c}."]
            if data in REMOTE_SENSING:
                 template = template = [lambda c: f"an aerial photograph of {c}."]
            return d, d.classes, template
        else:
            # We tokenize the caption datasets
            d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)
            return d
    else:
        if data == "SciCap":
            d = SciCap(os.path.join(path, "scicap"), split=split, transform=preprocess_fn)
            d = TokenizedDataset(d, **data_kwargs)

        elif data in ["Simpsons", "Simpsons-Captions"]:
            d = load_dataset("Norod78/simpsons-blip-captions", keep_in_memory=True)
            image_key, text_key = "image", "text"

            def transform(batch, MAXLEN=77):
                batch[image_key] = [preprocess_fn(image) for image in batch[image_key]]
                batch[text_key] = [text[:MAXLEN] for text in batch[text_key]]
                return batch
            d.set_transform(transform)

            train_ratio = 0.9  # use 90% for training data
            d_train, d_val = split_data(d["train"], train_ratio, seed=42, hf_data=True)
            d = d_train if is_train else d_val

            d = TokenizedDataset(d, image_key=image_key, text_key=text_key, **data_kwargs)

        elif data == "Simpsons-Images":
            d = ImageFolder("./data/kaggle_simpsons_characters/simpsons_dataset", transform=preprocess_fn)

        elif data =='FASHION-ALL':
            d = ConcatDataset([
                Fashion200k("./data/fashion200k", split=split, transform=preprocess_fn),
                # FashionGen("/data/FashionGen", split=split, transform=preprocess_fn),
                Polyvore("./data/polyvore", split=split, transform=preprocess_fn),
            ])
            d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)

        elif data == 'RS-ALL':
            d = ConcatDataset([
                RSICD("./data/RSICD", split=split, transform=preprocess_fn),
                UCMCaptions("./data/UCM_captions", split=split, transform=preprocess_fn),
                SydneyCaptions("./data/Sydney_captions", split=split, transform=preprocess_fn),
            ])
            d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)
        else:
            raise ValueError(f"Unknown dataset: {data}")

        return d


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        train_kwargs = {"is_train": True, "preprocess_fn": preprocess_train, "tokenizer": tokenizer}

        if args.keyword_path is not None:
            keywords = read_keywords(args.keyword_path)
            data["keyword"] = torch.cat([tokenizer(k) for k in keywords])
            train_kwargs.update({"keywords": keywords})

        if args.train_data == "RS-SHIFT":
            d_train = get_custom_data(args, "RS", **train_kwargs)
            d_train, _ = split_data(d_train, args.label_ratio, seed=args.seed)
            d_query, _, _ = get_custom_data(args, "RESISC45", **train_kwargs)
        elif args.train_data == "Simpsons":
            d_train = get_custom_data(args, "Simpsons-Captions", **train_kwargs)
            d_query = get_custom_data(args, "Simpsons-Images", **train_kwargs)
        else:
            d_train = get_custom_data(args, args.train_data, **train_kwargs)
            d_train, d_query = split_data(d_train, args.label_ratio, seed=args.seed)

        if args.method == "base":
            data["train"] = create_datainfo(args, d_train, args.batch_size, is_train=True)
        else:
            # assume L:U = 1:1
            data["train"] = create_datainfo(args, d_train, args.batch_size // 2, is_train=True)
            data["query"] = create_datainfo(args, d_query, args.batch_size // 2, is_train=True)

    if args.val_data:
        d_val = get_custom_data(args, args.val_data, preprocess_val, is_train=False, tokenizer=tokenizer)
        data["val"] = create_datainfo(args, d_val, args.batch_size, is_train=False)

    if args.imagenet_val is not None:
        d_zeroshot, classnames, template = get_custom_data(args, args.imagenet_val, preprocess_val, is_train=False)
        data["zeroshot-val"] = create_datainfo(args, d_zeroshot, args.batch_size, is_train=False)
        data["classnames"] = classnames
        data["template"] = template

    return data