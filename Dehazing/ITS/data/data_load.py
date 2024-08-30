import os
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, data='ITS', use_transform=True):
    image_dir = os.path.join(path, 'train_set')
    if data == 'ITS':
        crop_size = 256
    else:
        crop_size = 256

    data_transform = None
    if use_transform:
        data_transform = PairCompose(
            [
                PairRandomCrop(crop_size),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, data, transform=data_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, data, batch_size=1, num_workers=0):
    image_dir = path
    dataloader = DataLoader(
        DeblurDataset(image_dir, data, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, data, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'val_set'), data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, data, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(self.image_dir)
        self.groundtruth_path = os.path.join(os.path.dirname(self.image_dir), 'y')
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    
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

