# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import glob

def get_segmentation_masks_img(predictor, path):
    im = cv2.imread(path)
    outputs = predictor(im)
    return outputs['instances'].pred_masks.cpu()

def get_segmentation_masks(in_dir = '../data/VisWis_VQA_Grounding/val', out_path = '../processed/rcnn_segs.pth'):
    cfg = get_cfg()
    cfg.merge_from_file('../cache/mask_rcnn/mask_rcnn_X_101_32x8d_FPN_3x.yaml')
    cfg.MODEL.WEIGHTS = '../cache/mask_rcnn/model_final.pkl'
    predictor = DefaultPredictor(cfg)

    out = {}

    for img_path in glob.glob(f'{in_dir}/*'):
        mask = get_segmentation_masks_img(predictor, img_path)
        out[img_path] = mask

    print(f'#Images {len(out)}')
    torch.save(out, out_path)
    return

if __name__ == '__main__':
    get_segmentation_masks()
    # get_segmentation_masks(in_dir = '../data/VisWis_VQA_Grounding/test')
    print("done!!")