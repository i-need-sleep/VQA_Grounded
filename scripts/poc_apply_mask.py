import json
import os
from PIL import Image
import numpy as np
import cv2

DATA_DIR = '../data/VisWis_VQA_Grounding'
OUT_DIR = '../processed/masked'

def mask_pics(in_dir, mask_dir, out_dir):
    img_ids = os.listdir(in_dir)
    for img_id in img_ids:
        img_path = f'{in_dir}/{img_id}'
        mask_path = f'{mask_dir}/{img_id[:-3]}png'
        
        img = Image.open(img_path).convert('RGB')   
        mask = Image.open(mask_path).convert('RGB')   
        img = np.array(img)
        mask = np.array(mask)

        mask[mask > 0] = 1
        img = img * mask
        
        cv2.imwrite(f'{OUT_DIR}/{img_id[:-4]}.png', img)
    return

if __name__ == '__main__':
    mask_pics(f'{DATA_DIR}/train', f'{DATA_DIR}/binary_masks_png/train', OUT_DIR)
    print('done!!!')