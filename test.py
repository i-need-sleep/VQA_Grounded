from yaml import load
import torch
from utils.ocr_dataset import make_ocr_seqcls_loader, prep_input

a = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
print(a)
a = a.reshape(2,4)
print(a)

loader, tokenizer = make_ocr_seqcls_loader('VisWiz_VQA', 'train', 3)
for batch in loader:
    print(prep_input(batch, tokenizer))
    break