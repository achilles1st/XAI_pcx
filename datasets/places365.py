import torch
from torchvision.datasets import ImageFolder, Places365
from torchvision.transforms import transforms

from datasets.transforms import all_transforms


def get_places365(data_path, split="train", preprocessing=True):
    if split == "train":
        transform = all_transforms['imagenet']['val']
    else:
        transform = all_transforms['imagenet']['val']
    return Places365Dataset(data_path, split, transform=transform)


class Places365Dataset(Places365):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__(root, small=True, split="val", download=False, transform=transform)
        setattr(self, 'class_names', [str(i) for i in range(364)])
        setattr(self, 'num_classes', 364)
        self.preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_target(self, index: int) -> int:
        file, target = self.imgs[index]
        return target
