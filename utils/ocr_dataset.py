import json

from collections import Counter
from functools import lru_cache
from turtle import width

import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

DATA_ROOT = './datasets'
PROCESSED_ROOT = './processed'

class OCR_SeqCls_Dataset(Dataset):
    def __init__(self, dataset, split, thresh=0.4):
        self.dataset = dataset
        self.split = split
        self.annotation_path = f'{DATA_ROOT}/{dataset}/Annotations/{split}.json'
        self.ocr_path = f'{PROCESSED_ROOT}/ocr_{split}.json'
        self.thresh = thresh

        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            self.annotation = json.load(f)  

        with open(self.ocr_path, 'r', encoding='utf-8') as f:
            self.ocr = json.load(f)

        # Filter out images with no OCR results
        self.img_names = []
        for idx, annotation in enumerate(self.annotation):
            image_name = annotation['image']
            if len(self.ocr[image_name]) > 0:
                self.img_names.append([idx, image_name])

        print(f'{dataset} {split}')
        print(f'# images in the original dataset {len(self.annotation)}')
        print(f'# images with OCR results {len(self.img_names)}')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_idx, img_name = self.img_names[index]

        annotation = self.annotation[img_idx]
        ocr = self.ocr[img_name]

        question = annotation['question']
        answers = annotation['answers']

        # get the most popular answer
        answer = Counter([a['answer'] for a in answers]).most_common()[0][0]

        # get the image width and height
        h, w, _ = cv2.imread(f'{DATA_ROOT}/{self.dataset}/{self.split}/{img_name}').shape

        refed = []
        ref_coords = []

        for ocr_line in ocr:
            # normalize bbox coordinates
            coords = ocr_line[0]
            for coord in coords:
                coord[0] /= w
                coord[1] /= h
            ref_coords.append(coords)

            candidate = ocr_line[1][0]
            question += f'<mask>{candidate}'
            
            # count the candidate as positive if the substring levenshtein distance is small enough 
            lev = self._lev_dist(candidate.lower(), answer.lower())
            min_lev = len(answer) - len(candidate)
            if (lev - min_lev) / len(candidate) < self.thresh:
                refed.append(1)
            else:
                refed.append(0)
            
        return question, refed, ref_coords, img_name

    def _lev_dist(self, a, b):
        
        @lru_cache(None)  # for memorization
        def min_dist(s1, s2):

            if s1 == len(a) or s2 == len(b):
                return len(a) - s1 + len(b) - s2

            # no change required
            if a[s1] == b[s2]:
                return min_dist(s1 + 1, s2 + 1)

            return 1 + min(
                min_dist(s1, s2 + 1),      # insert character
                min_dist(s1 + 1, s2),      # delete character
                min_dist(s1 + 1, s2 + 1),  # replace character
            )

        return min_dist(0, 0)

def mr_collate(data):
    texts, coords, outs, img_names = [], [], [], []
    for line in data:
        texts.append(line[0])
        coords.append(line[2])
        outs += line[1]
        img_names.append(line[3])
    return texts, coords, outs, img_names

def make_ocr_seqcls_loader(dataset, split, batch_size):
    tokenizer = RobertaTokenizer.from_pretrained("./cache/roberta_large_tokenizer")
    dataset = OCR_SeqCls_Dataset(dataset, split)
    loader = DataLoader(dataset, batch_size=batch_size ,shuffle=True, collate_fn=mr_collate)
    return loader, tokenizer

def prep_input(batch, tokenizer):
    texts, coords, outs, img_names = batch
    tokenized = tokenizer(texts, padding=True)['input_ids']
    tokenized = torch.tensor(tokenized)
    
    cls_mask = tokenized == 50264

    coords_in = torch.zeros(cls_mask.shape[0], cls_mask.shape[1], 8)
    coords_ = []
    for coord in coords:
        coords_ += coord
    coords_ = torch.tensor(coords_).reshape(-1,8)
    coord_mask = cls_mask.unsqueeze(2)
    coord_mask = coord_mask.expand(-1, -1, 8)
    coords_in[cls_mask] = coords_

    outs = torch.tensor(outs)
    return tokenized, coords_in, cls_mask, outs, img_names