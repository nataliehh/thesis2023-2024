# Code based on: https://github.com/alinlab/s-clip/blob/master/custom/data.py
# And on: https://github.com/isaaccorley/torchrs/blob/main/torchrs/datasets/
import os
import h5py
import random
from PIL import Image
from multiprocessing import Value
from typing import List, Dict
import numpy as np
import gc
import logging

from tools import read_json

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler


# Active learning import
from open_clip import create_model_and_transforms

# Label parsing arguments
from textblob import TextBlob
import nltk
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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

class CustomDataLoader(torch.utils.data.Dataset):
    splits = ["train", "val", "test"]
    def __init__(self, root: str , split: str = 'train', transform: T.Compose = None, image_root: str = '', rawsent: bool = False, 
                 randomitem: bool = False, subclass: bool = False, cls: bool = False, kfold: int = -1, folds_folder: str = 'folds'):
        ''' Initialize custom data loader.
        root: path to the dataset folder with images + labels.
        split: which split to use (train, validation or test set).
        transform: type of image transformation to perform.
        image_root: the folder within root where the images are contained.
        rawsent: whether we should obtain the 'raw' sentences from item['sentences'].
        randomitem: whether we should pick a random item/image in get_item.
        subclass: whether to use the subclasses as opposed to the main classes of the dataset.
        cls: True if we want to use classification labels for y instead of captions.
        kfold: given a k-fold split of the data, which fold ([0,...,k-1]) to select.
        folds_folder: the location of the stored indices of the k-fold split of the data. 
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
        self.kfold = kfold
        self.folds_folder = folds_folder
        
    def __len__(self) -> int:
        return len(self.data)

    def use_kfold(self) -> bool: # whether to use K-fold cross validation
        folds_path = os.path.join(self.root, self.folds_folder)
        # Check if any folds are specified for this dataset (in that case, it has a 'folds' folder)
        if not os.path.exists(folds_path) or self.split == 'test': # don't use kfold split for the test data!
            return False
        folds = os.listdir(folds_path)
        # The folds have the format: split_fold_x.npy, where x is some integer that we extract here
        folds = [int(f.split('.')[0].split('_')[-1]) for f in folds]
        if self.kfold in folds: # If the specified fold has a corresponding fold index file, we use K-fold CV
            return True
        return False
        
    # Get the x value (= image) of an item, and transform it
    def get_x(self, item):
        key = 'image_path' if 'image_path' in item else 'filename'
        if self.randomitem and not self.cls:
            path = os.path.join(self.root, self.image_root, random.choice(item[key]))
        else:
            path = os.path.join(self.root, self.image_root, item[key])
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

class RS_CLS(CustomDataLoader): # For use with AID, WHU-RS19, NWPU-RESISC45, RSSCN7
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_class_info()
        
    def load_class_info(self):
        classes = [c for c in os.listdir(self.root) if os.path.isdir(os.path.join(self.root,c)) and 'numpy' not in c]
        data = []
        for class_ in classes:
            img_folder = os.path.join(self.root, class_)
            img_paths = os.listdir(img_folder)
            for img_path in img_paths:
                 data.append((img_path, class_))
        self.data = data
        self.classes = sorted(classes)
        self.classes_idx = {c:i for i, c in enumerate(self.classes)}
        
    def __getitem__(self, idx):
        img_path, class_ = self.data[idx]
        x = Image.open(os.path.join(self.root, class_, img_path)).convert('RGB')
        x = self.transform(x)
        y = self.classes_idx[class_]
        return x, y
            
        
# Data from: https://github.com/201528014227051/RSICD_optimal
class RSICD(CustomDataLoader):
    """ Data from paper: https://arxiv.org/abs/1712.07835 """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_anns = "dataset_rsicd.json" if 'RSICD' in self.root else "dataset.json"
        self.data = self.load_captions(os.path.join(self.root, data_anns), self.split)
        self.image_root = "RSICD_images"
        if self.cls and 'RSICD' in self.root:
            self.load_class_info(os.path.join(self.root, "txtclasses_rsicd"))

    def load_captions(self, path: str, split: str) -> List[Dict]:
        captions = read_json(path)["images"]
        if self.use_kfold():
            path = os.path.join(self.root, self.folds_folder, '{}_fold_{}.npy'.format(split, self.kfold))
            filenames = np.load(path)
            return [c for c in captions if c['filename'] in filenames]
            
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
            y = self.path2class[item['filename']]
            return x, y
        else:
            return super().__getitem__(idx)

class UCM(RSICD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.image_root = "images"

        if self.cls:
            self.load_class_info()
    
    def load_class_info(self):
        mapping = read_json(os.path.join(self.root,'caption_to_cls_mapping.json'))
        classes = sorted(set(mapping.values()))
        self.classes = classes
        self.classes_idx = {c:i for i, c in enumerate(self.classes)}
        self.path2class = mapping
    
    def __getitem__(self, idx):
        if self.cls:
            item = self.data[idx]
            x = self.get_x(item)
            str_label = self.path2class[item['filename']]
            y = self.classes_idx[str_label]
            return x, y
        else:
            return super().__getitem__(idx)

class SydneyCaptions(RSICD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.image_root = "images"

# Custom data collected by ILT's IDLab      
class ILT(CustomDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_root = "images"
        
        data_anns = "dataset.json"
        # Load all data
        self.data_all = read_json(os.path.join(self.root, data_anns))["images"]
        # Get the classes from all the data
        self.classes = sorted(set([c['label'] for c in self.data_all]))
        # Get the index of each class
        self.classes_idx = {c:i for i, c in enumerate(self.classes)}
        # Get the data for the specified split
        self.data = [c for c in self.data_all if c["split"] == self.split]
        
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        x = self.get_x(item) # Get the x value (= image)
        # If we want to do classification, we use the class names as labels for the images
        if self.cls:
            cls_name = item['label']
            y = self.classes_idx[cls_name]
            print(x, y)
            return x, y
        sentences = item["captions"] # y labels are the sentences of an image x
        
        return dict(x=x, captions=sentences)


# Data from: https://github.com/xthan/fashion-200k/tree/master
class Fashion200k(CustomDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self._load_annotation_db(self.split)

        if self.cls:
            # Remove some broken links
            self.data = [item for item in self.data if os.path.exists(os.path.join(self.root, item["image_path"]))]

            # Get a list of non-duplicate class names, and sort them
            self.classes = set([item['class_name'] for item in self.data])
            self.classes = sorted(self.classes)

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'test', 'test': 'test'}[split]

        # txt_path = ['dress', 'jacket', 'pants', 'skirt', 'top']
        # txt_format = 'labels/{}_{}_detect_all.txt' # Need to provide the fashion class and dataset split

        path = './data/fashion200k/labels'
        full_txt = []

        # Go over all text files
        # We concat all label files in the split to allow us to easily get all image paths and classes
        for txt in os.listdir(path):
            if split in txt: # If the text file contains the correct split ('train' or 'test')
                with open(os.path.join(path, txt), 'r') as f:
                    data = f.readlines()
                    full_txt += data

        image_paths = []
        descriptions = []
        # Extract the image paths from the labeling information files
        for item in full_txt:
            # The lines in the labels have the format: path\tconfidence_score\tdescription
            image_path, _, description = item.split('\t')
            image_path = image_path.split('/') # We extract the path to the image
            id_folder = image_path.pop(3) # Remove ID folder from path
            image_path = '/'.join(image_path) # Restore image path 
            image_paths.append(image_path)
            descriptions.append(description)

        if self.cls:
            class_index = 2 if self.subclass else 1

            data = []
            for image_path in image_paths:
                # The path has the format: /women/category/subcategory/image, we want either the category or subcategory
                class_name = image_path.split("/")[class_index].replace("_", " ") 
                data.append({ "image_path": image_path, "class_name": class_name })
        else:
            data = {}
            for i, image_path in enumerate(image_paths):
                # The names of items look like 12345_0.jpeg, here we extract the part before the underscore
                item_id = image_path.split('/')[-1].split('_')[0]

                if not os.path.exists(os.path.join(self.root, image_path)):
                    continue

                if item_id in data:
                    data[item_id]['image_path'].append(image_path)
                else:
                    data[item_id] = dict(image_path=[image_path], sentences=descriptions[i])
            data = [dict({'id': item_id}, **data[item_id]) for item_id in data]
        return data

# Data from: https://github.com/mvasil/fashion-compatibility
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
        # Read the file which contains the IDs belong to the (train/val/test) split
        split = {'train': 'train', 'val': 'valid'}[split]
        split_info_path = os.path.join(self.root, f"disjoint/{split}.json")
        split_info = read_json(split_info_path)
        # Make a list of all the image IDs belonging to the current split
        split_item_ids = []
        for grouping in split_info:
            items = grouping['items']
            item_ids = [item['item_id'] for item in items] # Get image IDs
            split_item_ids += item_ids
        split_item_ids = set(split_item_ids) # remove duplicates

        # Load in the item metadata, which contains names and descriptions of fashion items
        json_path = os.path.join(self.root, "polyvore_item_metadata.json")
        anno_json = read_json(json_path)
        item_ids = set(anno_json.keys()) # Get the (unique) IDs of all images

        id_intersection = item_ids.intersection(split_item_ids) # Keep only the IDs which are in the current split
        data = []

        for item_key in id_intersection:
            # Skip items not in the current split - calling this on anno_json is several magnitudes slower than the intersect call above
            # if item_key not in split_item_ids: 
            #     continue
            item = anno_json[item_key]
            item_path = item_key + '.jpg'
            if self.cls: # For classification, we use the (sub)category as the class label
                data.append({"image_path": item_path, "class_name": item["semantic_category"]}) 
            else: # For captions, we use the fashion item's title (=name) and description as its caption
                data.append({"image_path": item_path, "sentences": item["title"] + "." + item["description"]})

        return data

# Data from: https://github.com/monib110/fashionstyle-generation-GMM/blob/main/README.md
class FashionGen(CustomDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cls:
            self.data = self._load_annotation_db(self.split) 
            self.classes = set()
            for cls in self.data['input_category']:
                cls = cls[0].decode('UTF-8').lower()
                self.classes.add(cls)
            self.classes = list(sorted(list(self.classes)))
        else:
            self.data, self.images = self._load_annotation_db(self.split)
        
    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'validation'}[split]
        h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.h5") # hdf5
        h5_file = h5py.File(h5_path)
        if self.cls:
            return h5_file

        # Access the relevant lists for the for-loop
        h5_file_index = h5_file['index']
        h5_file_item_id = h5_file['input_productID']
        h5_file_input_name = h5_file['input_name']
        h5_file_input_desc = h5_file['input_description']
            
        data = {}
        for idx in range(len(h5_file_index)):
            item_id = int(h5_file_item_id[idx])
            input_name = h5_file_input_name[idx][0].decode('latin-1') + ". "
            input_desc = h5_file_input_desc[idx][0].decode('latin-1')    

            if item_id in data:
                data[item_id]['image_idx'].append(idx)
            else:
                data[item_id] = dict(image_idx=[idx], input_name=input_name, input_desc=input_desc)
        data = [dict({'id': item_id}, **data[item_id]) for item_id in data]

        images = h5_file['input_image']

        return data, images

    def __getitem__(self, idx):
        if self.cls:
            key = 'input_subcategory' if self.subclass else 'input_category'
            x = self.data['input_image'][idx]
            x = Image.fromarray(x)
            x = self.transform(x)
    
            cls_name = self.data[key][idx][0].decode('UTF-8').lower()
            y = self.classes.index(cls_name)
            return x, y
        item = self.data[idx]

        x = self.images[random.choice(item['image_idx'])]
        x = Image.fromarray(x)
        x = self.transform(x)

        sentences = item['input_name']
        sentences += item['input_desc']

        return dict(x=x, captions=sentences)

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
            caption = json_object['0-originally-extracted'] # Contains the text caption of the figure
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

@torch.no_grad()
def split_data(d, split_ratio, seed=42, hf_data=False, args = None, classnames = None, templates = None, model = None):
    if split_ratio == 1.0:
        return d, None
    gen = torch.Generator()
    gen.manual_seed(seed) # set random seed

    # The following parameters have to be configured for active learning to be possible:
    perform_AL = args.active_learning and args.train_data and templates is not None and classnames is not None

    # split labeled and unlabeled data
    perm_indices = torch.randperm(len(d), generator=gen)
    size = len(d) * split_ratio
    if perform_AL: # For AL, we request labels for label_ratio/total_epochs % of the data, each epoch
        current_size = int(size/args.al_iter)
        size = size * (args.current_iter+1)/args.al_iter
    size = int(size)
    if args.current_iter == 0: # Print the statistics of the labeled set and the full set of data
        # print('Dataset size:', len(d))
        logging.info(f'Labeled size: {size}')

    if perform_AL: # Active learning
        data = DataLoader(d,batch_size=128,shuffle=False,num_workers=0,pin_memory=True,sampler=None,)
        if model is None: # Load in base CLIP model if we don't have an existing model
            logging.info('Loading base CLIP...')
            model, _, _ = create_model_and_transforms(args.model, args.pretrained, precision=args.precision, 
                                                      device=args.device, output_dict=True, aug_cfg=args.aug_cfg,)
        chosen_idx = np.array([]) # Keep track of indices which have already been selected
        if args.current_iter > 0:
            chosen_idx = args.chosen_idx
        # Only allow indices to be picked if they're not already in our selected set
        possible_idx = np.array(list(set(perm_indices.tolist()) - set(chosen_idx)))
        possible_idx = np.sort(possible_idx) # Ensure the indices are sorted
        
        # Get the indices of the data in order of most uncertain to least uncertain
        # Either use ProbVLM or basic AL to get these indices
        if args.probvlm: 
            # Use the ProbVLM model only pre-trained on COCO if a fine-tuned variant isn't available yet
            probvlm_path = args.coco_resume_ckpt
            if args.current_iter != 0: # Otherwise, use the fine-tuned ProbVLM variant 
                probvlm_path = args.coco_save_ckpt  + '_last.pth' # The most recently saved fine-tuned variant
            uncertainty_idx = probvlm_AL(args, data, model, resume_path = probvlm_path)
        else: # Use basic active learning if we're not using ProbVLM
            uncertainty_idx = basic_active_learning(args, data, model, classnames, templates)

        # Keep only indices of images that are not labeled yet -> find overlap between the unlabeled idx and the uncertainty idx
        filter_for_possible_idx = torch.where(torch.isin(uncertainty_idx, torch.from_numpy(possible_idx).to(args.device))) 
        sorted_possible_idx = uncertainty_idx[filter_for_possible_idx] # Filter for the possible_idx indices

        # The indices are the labels with the highest uncertainty + already labeled indices
        indices = sorted_possible_idx.cpu().numpy()
        args.chosen_idx = np.append(chosen_idx, indices[:current_size])
        indices = np.append(chosen_idx, indices)
        
        # Clean up memory 
        del data, uncertainty_idx, possible_idx, filter_for_possible_idx, sorted_possible_idx, chosen_idx
        torch.cuda.empty_cache()
        gc.collect()
        
    else:
        indices = perm_indices
    
    if hf_data is False:
        d1 = Subset(d, indices[:size])
        d2 = Subset(d, indices[size:])
    else:
        d1 = [d[int(i)] for i in indices[:size]]
        d2 = [d[int(i)] for i in indices[size:]]

    # Clean up memory
    del indices
    torch.cuda.empty_cache()
    gc.collect()
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
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def format_for_template(classname, dataset):
    c = classname # Abbreviate for easier reading
    
    # Make UCM's labels consistent w/ RSICD's pascal case
    replacements = {'residential': 'Residential', 'mobilehomepark': 'MobileHomePark', 'tenniscourt': 'TennisCourt', 
                    'parkinglot': 'ParkingLot',  'baseballdiamond': 'BaseballDiamond', 'golfcourse': 'GolfCourse', 
                    'storagetank': 'StorageTank'}
    for old, new in replacements.items():
        c = c.replace(old, new)
    
    # RS classes use pascal case, split on the capitals (e.g. BareLand -> Bare Land)
    # From: https://stackoverflow.com/q/2277352/14467157
    c = re.sub( r"([A-Z])", r" \1", c)
    
    c = c.replace('&', 'and') # Replace '&' with 'and'
    c = c.replace(' and ', ' or ') # Replace ' and ' with ' or '
    
    c = TextBlob(c).words.singularize() # Make words singular (e.g. fields -> field)
    c = (' '.join(c)).lower() # Convert list to spaced string & make lowercase
    
    # Replace faulty singularization (e.g. 'ties' -> 'ty')
    # And keep spelling consistent (e.g. jewelry, not jewellery)
    replacements = {'ty': 'tie', 'jean': 'jeans', 'glass': 'glasses', 'legging': 'leggings', 'pant': 'pants', 
                    'bottom': 'bottoms', 'overpas': 'overpass', 'tenni': 'tennis', 'jewellery': 'jewelry'}
    for old, new in replacements.items():
        c = c.replace(old, new)

    # Mass nouns are nouns which don't use an article
    mass_nouns = set(['lingerie', 'jewelry', 'swimwear', 'underwear', 'outerwear', 'eyewear'])
    contains_mass_noun = len(mass_nouns.intersection(c.split())) > 0 
    
    # For RS labels like 'agricultural', 'residential', (i.e. not nouns) append 'area'
    if c.endswith('al') or c == 'parking' and 'Fashion' not in dataset: 
        c += ' area'
        
    # If the last word isn't plural (check the tag) nor a mass noun, add an article
    # .tags values have format (noun, tag), so we get the last index of the tuple
    if TextBlob(c).tags[-1][-1] != 'NNS' and not contains_mass_noun: 
        if c[0] in 'aeiou': # If a label's first word starts with a vowel, use 'an'
            c = 'an ' + c
        else: # Otherwise, use 'a' for the article
            c = 'a ' + c
    return c

# The links to most datasets were listed above, other datasets may be available via these scripts: https://github.com/isaaccorley/torchrs/tree/main/scripts
def get_custom_data(args, data, preprocess_fn, is_train, is_test = False, model = None, **data_kwargs):
    path = './data/'
    split = "train" if is_train else "test" if is_test else "val"
    if args.current_iter == 0:
        logging.info(f'{data} (split: {split})')
    cls = 'CLS' in data or data in ['AID', 'RESISC45', 'RSSCN7', 'WHU-RS19']
    subclass = 'SUBCLS' in data
    randomitem ='Fashion200k' in data

    data = data.replace('-CLS', '')
    data = data.replace('-SUBCLS', '')

    config = { # config dictionary that contains the call function to the dataset creator and the (relative) path to its data
        "RSICD": (RSICD, "RSICD", True),
        "UCM": (UCM, "UCM", True),
        "Sydney": (SydneyCaptions, "sydney_captions", False),
        "WHU-RS19": (RS_CLS, "WHU-RS19", True),
        "RSSCN7": (RS_CLS, "RSSCN7", True),
        "AID": (RS_CLS, "AID", True),
        "RESISC45": (RS_CLS, "NWPU-RESISC45", True),
        "Fashion200k": (Fashion200k, "fashion200k", True),
        "FashionGen": (FashionGen, "fashiongen", True), 
        "Polyvore": (Polyvore, "polyvore_outfits", True),
        "ILT": (ILT, "ILT_data", True),
    }
    REMOTE_SENSING = ["RSICD", "UCM", "Sydney", "RS.ALL", "WHU-RS19", "RSSCN7", "AID", "RESISC45"]

    # We use a dictionary for the configuration of each dataset loader
    if data in config:
        # A configuration specifies the class instantiation, path to the dataset and whether we are using a custom dataset loader
        # Custom dataset loaders allow for extra arguments
        dataset_class, dataset_path, custom = config.get(data) 
        
        if custom:
            d = dataset_class(os.path.join(path, dataset_path), split = split, transform=preprocess_fn, cls = cls, 
                              subclass = subclass, randomitem = randomitem, kfold = args.k_fold)
        else:
            d = dataset_class(os.path.join(path, dataset_path), transform=preprocess_fn)
        if cls:
            # Classif datasets use a captioning template, either 'a photo of [CLASS]' or 'an aerial photograph of [CLASS]' (RS data)
            template = [lambda c: f"a photo of {format_for_template(c, data)}."]
            if data in REMOTE_SENSING:
                 template = [lambda c: f"an aerial photograph of {format_for_template(c, data)}."]
            logging.info(f'CLS size: {len(d)}')
            return d, d.classes, template
        else:
            # We tokenize the caption datasets
            d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)
            return d
    else:
        if data =='Fashion.ALL':
            d = [Polyvore(os.path.join(path,"polyvore_outfits"), split=split, transform=preprocess_fn),
                Fashion200k(os.path.join(path,"fashion200k"), split=split, transform=preprocess_fn, randomitem = True),
                FashionGen(os.path.join(path,"fashiongen"), split=split, transform=preprocess_fn),]
            if args.current_iter == 0:
                d_lengths = [len(sub_d) for sub_d in d]
                logging.info('Sub-dataset sizes: {} (sum = {})'.format(d_lengths, sum(d_lengths)))
            d = ConcatDataset(d)    
            d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)

        elif data == 'RS.ALL':
            d = [RSICD(os.path.join(path,"RSICD"), split=split, transform=preprocess_fn, kfold = args.k_fold),
                UCM(os.path.join(path,"UCM"), split=split, transform=preprocess_fn, kfold = args.k_fold), # UCMCaption
                SydneyCaptions(os.path.join(path,"sydney_captions"), split=split, transform=preprocess_fn, kfold = args.k_fold),]
            if args.current_iter == 0:
                d_lengths = [len(sub_d) for sub_d in d]
                logging.info('Sub-dataset sizes: {} (sum = {})'.format(d_lengths, sum(d_lengths)))
            d = ConcatDataset(d)    
            d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)
        else:
            raise ValueError(f"Unknown dataset: {data}")
        return d


def get_data(args, preprocess_fns, iter=0, tokenizer=None, model=None):
    args.current_iter = iter
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    template, classnames = None, None

    if args.val_data:
        d_val = get_custom_data(args, args.val_data, preprocess_val, is_train=False, tokenizer=tokenizer)
        data["val"] = create_datainfo(args, d_val, args.batch_size, is_train=False)

    if args.imagenet_val is not None:
        d = get_custom_data(args, args.imagenet_val, preprocess_val, is_train=False)
        if len(d) == 3:
            d_zeroshot, classnames, template = d
        else: # Some datasets come in the format [(image, classname), (image, classname)], so we fix this
            d_zeroshot, classnames = zip(*d)
            template = [lambda c: f"an aerial photograph of {c}."]
        data["zeroshot-val"] = create_datainfo(args, d_zeroshot, args.batch_size, is_train=False)
        data["classnames"] = classnames
        data["template"] = template

    if args.imagenet_test is not None:
        d = get_custom_data(args, args.imagenet_test, preprocess_val, is_train=False, is_test = True)
        if len(d) == 3:
            d_zeroshot, classnames, template = d
        else: # Some datasets come in the format [(image, classname), (image, classname)], so we fix this
            d_zeroshot, classnames = zip(*d)
            template = [lambda c: f"an aerial photograph of {c}."]
        data["zeroshot-val"] = create_datainfo(args, d_zeroshot, args.batch_size, is_train=False)
        data["classnames"] = classnames
        data["template"] = template

    if args.train_data:
        train_kwargs = {"is_train": True, "preprocess_fn": preprocess_train, "tokenizer": tokenizer}

        if args.keyword_path is not None:
            keywords = read_keywords(args.keyword_path)
            data["keyword"] = torch.cat([tokenizer(k) for k in keywords])
            train_kwargs.update({"keywords": keywords})

        if args.train_data == "RS-SHIFT":
            d_train = get_custom_data(args, "RS", **train_kwargs)
            d_train, _ = split_data(d_train, args.label_ratio, seed = args.seed, args = args, model = model)
            d_query, _, _ = get_custom_data(args, "RESISC45", **train_kwargs)
        elif args.train_data == "Simpsons":
            d_train = get_custom_data(args, "Simpsons-Captions", **train_kwargs)
            d_query = get_custom_data(args, "Simpsons-Images", **train_kwargs)
        else:
            d_train = get_custom_data(args, args.train_data, **train_kwargs)
            d_train, d_query = split_data(d_train, args.label_ratio, seed=args.seed, args = args, model = model, 
                                          templates = template, classnames = classnames)

        if args.method == "base":
            data["train"] = create_datainfo(args, d_train, args.batch_size, is_train=True)
        else:
            # assume L:U = 1:1
            data["train"] = create_datainfo(args, d_train, args.batch_size // 2, is_train=True)
            data["query"] = create_datainfo(args, d_query, args.batch_size // 2, is_train=True)

    if args.test_data:
        test_kwargs = {"is_train": False, "is_test": True, "preprocess_fn": preprocess_train, "tokenizer": tokenizer}
        d_test = get_custom_data(args, args.test_data, **test_kwargs)
        data["test"] = create_datainfo(args, d_test, args.batch_size, is_train=False)
    
    return data

############################### ACTIVE LEARNING
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.functional import pairwise_cosine_similarity
from precision import get_autocast
from open_clip import get_cast_dtype, get_tokenizer

import sys

cwd = os.getcwd() # get current working directory
if 'tensusers' in cwd:
    sys.path.append('/vol/tensusers4/nhollain/ProbVLM/src') # Allow probvlm imports
    sys.path.append('/vol/tensusers4/nhollain/ProbVLM/') # Allow probvlm imports

    from utils import get_features_uncer_ProbVLM, sort_wrt_uncer
else:
    from probvlm.utils import get_features_uncer_ProbVLM, sort_wrt_uncer
    sys.path.append('./probvlm/')
    sys.path.append('./probvlm/src')
    
from networks import get_default_BayesCap_for_CLIP

@torch.no_grad()
def basic_active_learning(args, data, model, classnames = None, templates = None):
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
    
    # Clean memory
    del sim, entropy, uncertainty, image_features, text_features, data
    torch.cuda.empty_cache()
    gc.collect()
    return idx_by_uncertainty

@torch.no_grad()
def probvlm_AL(args, data, model, resume_path = ''):
    CLIP_Net = model.to('cuda') 
    # Load the pre-trained ProbVLM adapter 
    ProbVLM_Net = get_default_BayesCap_for_CLIP()
    checkpoint = torch.load(resume_path)
    ProbVLM_Net.load_state_dict(checkpoint['model'])

    # Get a dictionary of the probabilistic parameters 
    r_dict = get_features_uncer_ProbVLM(CLIP_Net, ProbVLM_Net, data, dictionary = True)
    
    # Get the (sorted) uncertainties for the images (discard text uncertainty)
    uncertainty_images, _ = sort_wrt_uncer(r_dict)
    # The uncertainties are given as (idx, uncertainty), we extract only the idx
    idx_by_uncertainty = torch.Tensor([u[0] for u in uncertainty_images]).to(args.device)
    
    # Clean memory
    del checkpoint, r_dict, ProbVLM_Net, data
    torch.cuda.empty_cache()
    gc.collect()
    return idx_by_uncertainty
