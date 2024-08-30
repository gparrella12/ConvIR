import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_dataloader(path, batch_size=64, num_workers=0):
    image_dir = os.path.join(path, 'train_set')

    dataloader = DataLoader(
        DeblurDataset(image_dir, ps=256),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = path
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'val_set'), is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

import random
class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(self.image_dir)
        self.groundtruth_path = os.path.join(os.path.dirname(self.image_dir), 'y')
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        groundtruth_name = image_name.split('_')[-1]
        image = Image.open(os.path.join(self.image_dir, image_name))
        label = Image.open(os.path.join(self.groundtruth_path, groundtruth_name))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label



    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
