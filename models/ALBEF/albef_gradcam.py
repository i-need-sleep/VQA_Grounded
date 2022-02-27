import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from PIL import Image

from functools import partial
from models.model_vqa import ALBEF
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from models.tokenization_bert import BertTokenizer
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

import torch
from torch import nn
from torchvision import transforms

import cv2
from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

CONFIG_PATH = './configs/VQA_viswis.yaml'
CACHE_DIR = '../../cache/'
CHECKPOINT_PATH = '../../output/vqa/epo13_saved.pth'
GT_PATH = '../../data/VisWis_VQA_Grounding/annotations/val_grounding.json'
GT_PATH_TEST = '../../data/VisWis_VQA_Grounding/annotations/test_grounding.json'
GT_MASK_DIR = '../../data/VisWis_VQA_Grounding/binary_masks_png/val/'
IMG_DIR = '../../data/VisWis_VQA_Grounding/val/'
IMG_DIR_TEST = '../../data/VisWis_VQA_Grounding/val/test/'
SEG_PATH = '../../processed/rcnn_segs.pth'
SEG_PATH = '../../processed/rcnn_segs_TEST.pth'
OUT_DIR = '../../output/albef_gradcam/'
OUT_DIR_TEST = '../../output/albef_gradcam_test/'
BLOCK_NUM = 8
THRESH = 0.5

def getAttMap(img, attMap, blur = True, overlap = False):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

def albef_gradcam(val = True):
    # Setup
    config = yaml.load(open(CONFIG_PATH, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Original image paths
    if val:
        with open(GT_PATH, 'r') as f:
            trues_data = json.load(f)
    else:
        with open(GT_PATH_TEST, 'r') as f:
            trues_data = json.load(f)
    trues = []
    for key, val in trues_data.items():
        trues.append(key)

    # Dataset and tokenizer
    print("Creating vqa datasets")
    datasets = create_dataset('vqa_viswis', config)   
    
    samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 

    text_enc = f'{CACHE_DIR}bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(text_enc)

    # Load model
    model = ALBEF(config=config, text_encoder=text_enc, text_decoder=text_enc, tokenizer=tokenizer)
    model = model.to(device)   

    model.text_encoder.base_model.base_model.encoder.layer[BLOCK_NUM].crossattention.self.save_attention = True

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu') 
    state_dict = checkpoint['model']

    msg = model.load_state_dict(state_dict)  
    print('load checkpoint from %s'%CHECKPOINT_PATH)
    print(msg)

    answer_list = [answer+config['eos'] for answer in test_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)   

    # Load seg data
    if val:
        seg_data = torch.load(SEG_PATH, map_location=torch.device('cpu'))
    else:
        seg_data = torch.load(SEG_PATH_TEST, map_location=torch.device('cpu'))
    
    IoUs = []
    # Compute gradcam
    for idx, (image, question, question_id) in enumerate(test_loader):
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs, logits = model(image, question_input, answer_input, train=False, k=config['k_test'], return_logits=True)      
        max_logit = torch.max(logits, 1, keepdim=True).values
        loss = max_logit[:,0].sum()

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            mask = question_input.attention_mask.view(question_input.attention_mask.size(0),1,-1,1,1)

            grads=model.text_encoder.base_model.base_model.encoder.layer[BLOCK_NUM].crossattention.self.get_attn_gradients()
            cams=model.text_encoder.base_model.base_model.encoder.layer[BLOCK_NUM].crossattention.self.get_attention_map()

            cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 24, 24) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 24, 24) * mask

            gradcam = cams * grads
            gradcam = gradcam[0].mean(0).cpu().detach()
        
        # Fetch original image
        img_id = trues[idx]
        if val:
            img_path = f'{IMG_DIR}{img_id}'
        else:
            img_path = f'{IMG_DIR_TEST}{img_id}'

        rgb_image = cv2.imread(img_path)[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255

        # Get attn map
        
        for i in range(gradcam.shape[0]-1):
            if i == 0:
                img_out = getAttMap(rgb_image, gradcam[i+1])
            else:
                img_out += getAttMap(rgb_image, gradcam[i+1])
        img_out /= gradcam.shape[0]-1

        activated = img_out > THRESH
        inactivated = img_out <= THRESH
        img_out[activated] = 1
        img_out[inactivated] = 0

        # Fetch segmentations
        if 'test' in img_id:
            key = '../data/VisWis_VQA_Grounding/test/' + img_id
        else:
            key = '../data/VisWis_VQA_Grounding/val/' + img_id
        segs = seg_data[key]
        img_out = torch.tensor(img_out).to(device).unsqueeze(0)
        segs = segs.to(device)
        seg_mask = segs * img_out

        seg_mask = torch.sum(seg_mask, 1)
        seg_mask = torch.sum(seg_mask, 1)
        seg_mask[seg_mask > 0] = 1
        seg_mask = seg_mask.unsqueeze(1)
        seg_mask = seg_mask.unsqueeze(1)
        
        mask = segs * seg_mask 
        
        mask = torch.sum(mask, axis=0)
        mask[mask > 0] = 255

        # Save the image
        mask_out = mask.cpu().detach().numpy()
        mask_out = np.expand_dims(mask_out, axis=2)
        mask_out = np.repeat(mask_out, 3, axis=2)
        if val:
            cv2.imwrite(f'{OUT_DIR}{img_id[:-4]}.png', mask_out)
        else:
            cv2.imwrite(f'{OUT_DIR_TEST}{img_id[:-4]}.png', mask_out)

        if val:
            # Fetch GT mask
            mask[mask > 0] = 1
            gt_path = f'{GT_MASK_DIR}{img_id[:-4]}.png'
            img = cv2.imread(gt_path)[:,:,0]
            img = torch.tensor(img)
            img[img>0] = 1

            img = img.to(device)
        
            # Get IOU
            I = img * mask
            U = img + mask
            U[U > 0] = 1
            IoU = torch.sum(I) / torch.sum(U)
            IoUs.append(IoU.cpu().item())
            print(gt_path)
            # break

    if val:
        print(sum(IoUs)/len(IoUs))
    return

if __name__ == '__main__':
    albef_gradcam()
    albef_gradcam(val=False)
    print('Done!!!')