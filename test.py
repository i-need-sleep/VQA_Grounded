import torch


device = torch.device('cpu')
SEG_PATH = './processed/rcnn_segs.pth'

seg_data = torch.load(SEG_PATH, map_location=device)
for key, val in seg_data.items():
    print(key)
    break

print("done!!")