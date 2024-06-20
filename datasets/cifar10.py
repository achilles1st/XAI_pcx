import os

import torch.nn
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity

from datasets.transforms import all_transforms


def get_cifar10(data_path="/hardd/datasets/cifar10", split="train", preprocessing=True):
    if split == "train":
        transform = all_transforms['cifar10']['train']
    else:
        transform = all_transforms['cifar10']['val']
    return create_datasets(data_path, split, transform=transform)


def create_datasets(root, split='test', transform=None):

    if split == 'train':
        data = datasets.CIFAR10(root=root, train=True, transform=transform, download=False)
    elif split == 'val' or split == 'test':
        data = datasets.CIFAR10(root=root, train=False, transform=transform)

    setattr(data, 'class_names', list(idx_to_class.values()))
    setattr(data, 'num_classes', len(idx_to_class.values()))
    setattr(data, 'preprocessing', torch.nn.Identity())

    def get_target(self, idx):
        return self.targets[idx]

    setattr(data, 'get_target', get_target.__get__(data))



    return data


idx_to_class = {0: 'airplane',
                1: 'automobile',
                2: 'bird',
                3: 'cat',
                4: 'deer',
                5: 'dog',
                6: 'frog',
                7: 'horse',
                8: 'ship',
                9: 'truck'}
