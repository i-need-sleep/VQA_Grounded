import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question


class vqa_viswis_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, eos='[SEP]', split="train", max_ques_words=30, answer_list=''):
        self.split = split        
        self.ann = []
        with open(ann_file[0], 'r') as f:
            data = json.load(f)
        for key, val in data.items():
            val['image'] = key
            self.ann.append(val)

        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            self.answer_list = json.load(open(answer_list,'r'))   
                
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.vqa_root, ann['image'])  
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split == 'test':
            question = pre_question(ann['question'],self.max_ques_words)
            return image, question, index


        elif self.split=='train':                       
            
            question = pre_question(ann['question'],self.max_ques_words)        
                
            answer_weight = {}
            for answer in ann['answers']:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1/len(ann['answers'])
                else:
                    answer_weight[answer] = 1/len(ann['answers'])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            answers = [answer+self.eos for answer in answers]
                
            return image, question, answers, weights