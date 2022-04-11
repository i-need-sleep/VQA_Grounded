from yaml import load
import torch
from utils.ocr_dataset import make_ocr_seqcls_loader, prep_input
from models.ocr_seqcls import OCR_seqcls_model

model = OCR_seqcls_model()
loader, tokenizer = make_ocr_seqcls_loader('VisWiz_VQA', 'train', 3)
for batch in loader:
    tokenized, coords_in, cls_mask, outs, img_names = prep_input(batch, tokenizer)
    print(cls_mask)
    out = model(tokenized, coords_in, cls_mask)
    print(out)
    print(out.shape)
    break