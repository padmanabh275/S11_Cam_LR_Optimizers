import torch
import torchvision
import numpy as np
import albumentations as A
from torchvision import datasets, transforms
from albumentations.pytorch import ToTensorV2

# Pixel statistics of all (train + test) CIFAR-10 images
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
means = (0.4914, 0.4822, 0.4465) # Mean
stds= (0.2023, 0.1994, 0.2010) # Standard deviation
chw = (3, 32, 32) # Channel, height, width
CLASSES = [ # Class labels (list index = class value)
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]
class Cifar10SearchDataset(torch.utils.data.Dataset):
    def __init__(self,
        dataset:torchvision.datasets,
        transform:A.Compose) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        if self.transform: image = self.transform(image=np.array(image))['image']
        return image, label

def transformed_data()-> dict[str, A.Compose]:
    return {
        'train_transform': A.Compose([
            A.Normalize(mean=means, std=stds, always_apply=True),
            A.PadIfNeeded(min_height=4, min_width=4, always_apply=True),
            A.RandomCrop(height=32, width=32,always_apply=True),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=means
            ),
            ToTensorV2(),
        ]),
    
        'test_transform':A.Compose([
            A.Normalize(mean=means, std=stds, always_apply=True),
            ToTensorV2(),
        ]),
    
    }

def get_data(transform:dict[str, A.Compose],) -> dict[str, Cifar10SearchDataset]:
    "Create training and test datasets and apply transforms."
    return {
        'train': Cifar10SearchDataset(
            dataset=torchvision.datasets.CIFAR10('../data', train=True, download=True),
            transform=transform['train_transform'],
        ),
        'test': Cifar10SearchDataset(
            dataset=torchvision.datasets.CIFAR10('../data', train=False, download=False),
            transform=transform['test_transform'],
        ),
    }

def get_dataloader(
    dataset:Cifar10SearchDataset,
    params:dict[str, bool or int],
) -> dict[str, torch.utils.data.DataLoader]:
    "Create training and test dataloader."
    return {
        'train': torch.utils.data.DataLoader(dataset['train'], **params),
        'test': torch.utils.data.DataLoader(dataset['test'], **params),
    }
 