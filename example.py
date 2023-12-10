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

ROOT = Path(__file__).parent
IMAGES = ROOT/'images'
DATASET = ROOT/'dataset'
# TEST = DATASET/'test'
TEST = ROOT/'validation'
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
# target_transform = v2.Compose([
#     v2.Resize(img_size)
#                       ])
target_transform = None
training_data = SegmentationDataset(TEST, device, augment=True, transform=transform, target_transform=target_transform)
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

def get_px(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{(x,y)}, {seg_np[y, x]}, {name_list[seg_np[y, x]]}")

data_iterator = iter(train_dataloader)
while True:
    x, labels = next(data_iterator)
    # img_np = x[0].detach().permute(1, 2, 0).cpu().numpy()
    img_np = torch.flip(x[0], (2,)).detach().permute(1, 2, 0).cpu().numpy()
    img_np = img_np.astype(np.uint8)
    output = model(x)
    output = torch.flip(output, (3,))
    seg_np = output.argmax(dim=1)[0].detach().cpu().numpy()
    seg_img = colorize(seg_np) # <---- input is numpy, output is PIL.Image
    cv2.imshow('img', (np.asarray(seg_img)//3 + (cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)//3)*2))
    # cv2.imshow('img', (cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)))
    cv2.setMouseCallback('img',get_px)
    cv2.waitKey(0)
    cv2.destroyWindow('img')
    # break
# Function from fastseg to visualize images and output segmentation

# blended_img = blend(img, seg_img) # <---- input is PIL.Image in both arguments

# Concatenate images for simultaneous view
# new_array = np.concatenate((np.asarray(blended_img), np.asarray(seg_img)), axis=1)

# Show image from PIL.Image class
# combination = Image.fromarray(new_array)
# combination.show()

# print(name_list)