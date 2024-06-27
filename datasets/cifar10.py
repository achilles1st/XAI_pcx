import os
import torch.nn
from torchvision import datasets, transforms

from datasets.transforms import all_transforms


class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CustomCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.class_names = list(idx_to_class.values())
        self.num_classes = len(idx_to_class.values())
        self.preprocessing = torch.nn.Identity()

    def get_target(self, idx):
        return self.targets[idx]

    # To make the  script compatible with num_workers greater than 0, we'll modify the dataset
    # class to ensure it can be pickled. This involves defining the __getstate__ and __setstate__ methods,
    # which handle the pickling and unpickling of the dataset object.
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def get_cifar10(data_path="C:/Users/tosic/XAI/pcx/datasets", split="train", preprocessing=True):
    if split == "train":
        transform = all_transforms['cifar10']['train']
    else:
        transform = all_transforms['cifar10']['val']
    return create_datasets(data_path, split, transform=transform)


def create_datasets(root, split='test', transform=None):
    if split == 'train':
        data = CustomCIFAR10(root=root, train=True, transform=transform, download=False)
    elif split == 'val' or split == 'test':
        data = CustomCIFAR10(root=root, train=False, transform=transform)
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
