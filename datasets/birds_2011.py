import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as T


class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if np.random.choice([0, 1]):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



def get_birds(data_path, preprocessing=False, split="train"):
    return BirdDataset(f"{data_path}", f"{data_path}/images", f"{data_path}/segmentations",
                       train=split == "train", segmentation=False, preprocessing=preprocessing)


class BirdDataset(Dataset):

    def __init__(self, image_paths, image_dir, segmentation_dir, train: bool = True, segmentation: bool = True,
                 preprocessing: bool = False):
        super(BirdDataset, self).__init__()
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.preprocessing = T.Normalize([0.47473491, 0.48834997, 0.41759949], [0.22798773, 0.22288573, 0.25982403])
        if preprocessing:
            transforms_image = transforms.Compose(
                [T.Resize(224), T.CenterCrop(224), T.ToTensor(), AddGaussianNoise(0, 0.05),
                 T.RandomHorizontalFlip(), T.RandomAffine(10, (0.2, 0.2), (0.8, 1.2)),
                 self.preprocessing])
        else:
            transforms_image = transforms.Compose(
                [T.Resize(224), T.CenterCrop(224), T.ToTensor(), AddGaussianNoise(0, 0.05),
                 T.RandomHorizontalFlip(), T.RandomAffine(10, (0.2, 0.2), (0.8, 1.2))])
        if preprocessing:
            transforms_image_test = transforms.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                self.preprocessing])
        else:
            transforms_image_test = transforms.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor()
            ])

        self.transform_image = transforms_image_test if train else transforms_image_test
        self.transform = self.transform_image
        self.segmentation = segmentation
        self.path = image_paths
        with open(image_paths + '/images.txt', 'r') as f:
            self.images_paths = [line.split(" ")[-1] for line in f.readlines()]
        with open(image_paths + '/image_class_labels.txt', 'r') as f:
            self.class_ids = [int(line.split(" ")[-1][:-1]) for line in f.readlines()]
        with open(image_paths + '/train_test_split.txt', 'r') as f:
            self.train_split = np.array([int(line.split(" ")[-1][:-1]) for line in f.readlines()])
        with open(image_paths + '/classes.txt', 'r') as f:
            self.class_names = np.array([str(line.split(" ")[-1][:-1].split(".")[-1]) for line in f.readlines()])

        # for i, class_name in enumerate(self.class_names):
        #     print(i, class_name)
        if train:
            self.class_ids = np.array(self.class_ids)[self.train_split == 0]
            self.images_paths = np.array(self.images_paths)[self.train_split == 0]
            self.img_ids = np.nonzero(self.train_split)[0] + 1
        else:
            self.class_ids = np.array(self.class_ids)[self.train_split == 1]
            self.images_paths = np.array(self.images_paths)[self.train_split == 1]
            self.img_ids = np.nonzero(1 - self.train_split)[0] + 1

        # setattr(self, 'class_names', [str(i) for i in range(200)])
        setattr(self, 'num_classes', 200)

    def get_target(self, idx):
        return self.class_ids[idx] - 1
    def __getitem__(self, index):
        image_name = ".".join(self.images_paths[index].split('.')[:-1])

        image = Image.open(os.path.join(self.image_dir, f"{image_name}.jpg")).convert("RGB")


        image = self.transform_image(image)
        return image, self.class_ids[index] - 1

    def __len__(self):
        return len(self.images_paths)

    @staticmethod
    def reverse_augmentation(data: torch.Tensor) -> torch.Tensor:
        data = data.clone() + 0
        mean = torch.Tensor((0.485, 0.456, 0.406)).to(data)
        var = torch.Tensor((0.229, 0.224, 0.225)).to(data)
        data *= var[:, None, None]
        data += mean[:, None, None]
        return torch.multiply(data, 255)


def data_transforms(data: torch.Tensor) -> torch.Tensor:
    mean = torch.Tensor((0.485, 0.456, 0.406)).to(data)
    var = torch.Tensor((0.229, 0.224, 0.225)).to(data)
    # mean = torch.Tensor((0.47473491, 0.48834997, 0.41759949)).to(data)
    # var = torch.Tensor((0.22798773, 0.22288573, 0.25982403)).to(data)
    data *= var[:, None, None]
    data += mean[:, None, None]
    return torch.multiply(data, 255)


def label_img_to_color(img):
    ind = np.arange(201)
    rgb = [(r, r, r) for r in np.arange(201) / 200 * 255]
    label_to_color = dict(zip(ind, rgb))

    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])

    return img_color
