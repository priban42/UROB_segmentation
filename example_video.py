import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from fastseg import MobileV3Small
import cv2
from fastseg.image.colorize import colorize, blend
from pathlib import Path
from SegmentationDataset import SegmentationDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision.transforms.functional as F

ROOT = Path(__file__).parent
VIDEO = ROOT/'vid1.mp4'
# TEST = ROOT/'validation'
device = torch.device("cuda")

model = MobileV3Small.from_pretrained()
# model = MobileV3Small(num_classes=2)
# model.load_state_dict(torch.load(ROOT/"model_1.pth"))
# model.load_state_dict(torch.load(ROOT/"model_2.pth"))
model.load_state_dict(torch.load(ROOT/"model_3.pth"))
model.to(device)
name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
img_size = (512, 910)

transform = v2.Compose([
    v2.Resize(img_size),
    v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
                      ])


def square_pad(image):
    h, w = image.shape[-2:]
    max_wh = np.max([w // 2, h])
    hp = int((max_wh - w // 2) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, 1, 'constant')
# target_transform = v2.Compose([
#     v2.Resize(img_size)
#                       ])
target_transform = None

writer= cv2.VideoWriter('output_vid.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (img_size[1], img_size[0]))

def img_to_tensor(image):
    x = torch.tensor(image, device=device, dtype=torch.float32).unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    return x

cam = cv2.VideoCapture(str(VIDEO))
while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break
    x = img_to_tensor(frame)
    x = square_pad(x)
    x = transform(x)
    # img_np = x[0].detach().permute(1, 2, 0).cpu().numpy()
    img_np = torch.flip(x[0], (2,)).detach().permute(1, 2, 0).cpu().numpy()
    img_np = img_np.astype(np.uint8)
    output = model(x)
    # output = torch.flip(output, (3,))
    seg_np = output.argmax(dim=1)[0].detach().cpu().numpy()
    seg_img = colorize(seg_np) # <---- input is numpy, output is PIL.Image
    disp_frame = np.asarray(seg_img)//4 + (img_np//4)*3
    writer.write(disp_frame)
    cv2.imshow('img', (disp_frame))
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    cv2.imshow('img', (cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))

cam.release()
writer.release()
cv2.destroyAllWindows()
