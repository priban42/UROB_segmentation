import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
from torchvision.transforms import v2
import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 1, 'constant')



class SegmentationDataset(Dataset):
    def __init__(self, dataset_path, device, augment=False, transform=None, target_transform=None):
        self.device = device
        self.DATASET = Path(dataset_path)
        self.RGB = self.DATASET / 'rgb'
        self.SEG = self.DATASET / 'seg'
        self.images = sorted(os.listdir(self.RGB))
        self.labels = sorted(os.listdir(self.SEG))
        assert len(self.images) == len(self.labels), f"len(self.train_img): {len(self.images)}, len(self.train_labels): {len(self.labels)}"
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment

    def square_pad(self, image):
        h, w = image.shape[-2:]
        max_wh = np.max([w//2, h])
        hp = int((max_wh - w//2) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 1, 'constant')

    def __len__(self):
        return len(self.images)*100
    #
    # def __load_img(path):
    #     img = Image.open(path).resize((910, 512))
    #     img_np = np.asarray(img)
    #     x = torch.tensor(img_np, device=device).unsqueeze(0).float()
    #     x = x.permute(0, 3, 1, 2)
    #     x = (x / 255) * 2 - 1
    #     return x

    def __getitem__(self, idx):
        idx_mod = idx%len(self.images)
        image = read_image(str(self.RGB / self.images[idx_mod]))
        # image = image.permute(0, 2, 1)
        image = image.to(self.device, dtype=torch.float32)
        # image = (image / 255) * 2 - 1


        label = np.load(self.SEG / self.labels[idx_mod])
        label[label > 0] = 1
        label = torch.tensor(label, device=self.device).long()
        # image = self.transform(image)
        image = self.square_pad(image)
        label = self.square_pad(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.augment:
            if bool(random.getrandbits(1)):
                image = torch.flip(image, (2,))
                label = torch.flip(label, (1,))
        return image, label

