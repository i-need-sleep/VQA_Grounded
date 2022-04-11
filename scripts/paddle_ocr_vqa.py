import os
import json
import tqdm

from PIL import Image

from paddleocr import PaddleOCR,draw_ocr

IMG_ROOT = '../datasets/VisWiz_VQA/'
SPLIT = 'train'

def get_ocr(ocr, img_path, visualize=False):
    result = ocr.ocr(img_path, cls=True) # [[[[x1,y1],... [x4,y4]], ('text', confidence)], ]

    # draw result
    if visualize:
        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores)
        im_show = Image.fromarray(im_show)
        im_show.save('../result.jpg')

    return result

if __name__ == '__main__':
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    out = {}
    for img in tqdm.tqdm(os.listdir(f'{IMG_ROOT}{SPLIT}')):
        result = get_ocr(ocr, f'{IMG_ROOT}{SPLIT}/{img}')
        out[img] = result
    
    for img, val in out.items():
        for line in val:
            for idx, coord in enumerate(line[0]):
                line[0][idx][0] = float(line[0][idx][0])
                line[0][idx][1] = float(line[0][idx][1])
            line[1] =  (line[1][0], float(line[1][1]))

    with open(f'../processed/ocr_vqa_{SPLIT}.json', 'w') as f:
        json.dump(out, f)