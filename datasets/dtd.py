import torch
from torchvision.datasets import ImageFolder


def get_dtd(data_path, split="train", preprocessing=True):
    if split == "train":
        transform = None
    else:
        transform = None
    return Textures(data_path, split, transform=transform)


class Textures(ImageFolder):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        setattr(self, 'class_names', list())
        setattr(self, 'num_classes', 0)
        setattr(self, 'preprocessing', torch.nn.Identity())
