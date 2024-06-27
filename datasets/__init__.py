from typing import Callable

from datasets.cifar10 import get_cifar10
from datasets.dtd import get_dtd
from datasets.imagenet import get_imagenet
# from datasets.isic import get_isic
from datasets.isun import get_isun
from datasets.lsun import get_lsun
from datasets.lsunr import get_lsunr
from datasets.places365 import get_places365
from datasets.svhn import get_svhn
from datasets.birds_2011 import get_birds

DATASETS = {
    "imagenet": get_imagenet,
    "birds": get_birds,
    "cifar10": get_cifar10,
    "svhn": get_svhn,
    "dtd": get_dtd,
    "lsun": get_lsun,
    "lsunr": get_lsunr,
    "isun": get_isun,
    # "isic": get_isic,
    "places365": get_places365,
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        print(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")