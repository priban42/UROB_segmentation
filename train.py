import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from fastseg import MobileV3Small
from fastseg.image.colorize import colorize, blend
import torchvision.transforms as T
from pathlib import Path
print("torch.cuda.is_available():", torch.cuda.is_available())
import time
import os
from SegmentationDataset import SegmentationDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2

device = torch.device("cuda")
# device = torch.device("cpu")

ROOT = Path(__file__).parent
# IMAGES = ROOT/'rgb'
# SEG = ROOT/'seg'
OUTPUT = ROOT/'output'

DATASET = ROOT/'dataset'
TRAIN = DATASET/'train'
TEST = DATASET/'train'



model = MobileV3Small.from_pretrained()
# model = MobileV3Small(num_classes=2)

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
training_data = SegmentationDataset(TRAIN, device, transform=transform, target_transform=target_transform)
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
# model = MobileV3Small(num_classes=2)

# Set up model to training mode (some layers are designed to behave differently during learning and during inference - batch norm for example.)
# Always learn model in training mode
model.train()

# def load_img(path):
#     img = Image.open(path).resize((910, 512))
#     img_np = np.asarray(img)
#     x = torch.tensor(img_np, device=device).unsqueeze(0).float()
#     x = x.permute(0, 3, 1, 2)
#     x = (x / 255) * 2 - 1
#     return x
#
# def load_labels(path):
#     labels = np.load(path)
#     labels[labels > 0] = 1
#
# def load_data(indeces = (1, 2, 3), train=True):
#
#     if train:
#         for i in indeces:
#
#     pass


# Set up optimizer to automatically update weights with respect to computed loss and negative of gradient
# Regularization weight decay - analogy with remembering the exam questions
CE = torch.nn.CrossEntropyLoss(reduction="none", weight=None)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# img = Image.open('rgb/0020.png').resize((910, 512))
# img_np = np.asarray(img)
# x_example = torch.tensor(img_np, device=device).unsqueeze(0).float()
# x_example = x_example.permute(0, 3, 1, 2)
# x_example = (x_example / 255) * 2 - 1
# # Multiple iterations (epochs)
# labels_example = np.load(TRAIN / 'seg' / '0020.npy')
# labels_example[labels_example > 0] = 1
# labels_example = torch.tensor(labels_example, device=device).unsqueeze(0).long()
start_time = time.time()
data_iterator = iter(train_dataloader)
for e in range(500):

    # Forward pass of model. Input image x and output as per-pixel probabilities, per-image in batch
    # output dimensions: Batch Size x Class probs x H x W
    x, labels = next(data_iterator)
    output = model(x)

    loss = CE(output, labels)

    print(f'Epoch: {e:03d}', f'Loss: {loss.mean().item():.4f}')
    loss.mean().backward()
    optimizer.step()

    # Test if the models has accumulated gradients and therefore "learn something"
    if e == 0:
        print("Gradient in the last layer on specific weights: ", model.last.weight[0, 0, 0, 0])

    optimizer.zero_grad()

    seg_np = output.argmax(dim=1)[0].detach().cpu().numpy()
    seg_img = colorize(seg_np)
    seg_img.save(OUTPUT/f'{e:03d}.png')


print(f"elapsed time: {time.time() - start_time:.02f}s")
# Saving weights
torch.save(model.state_dict(), ROOT/'model_3.pth')