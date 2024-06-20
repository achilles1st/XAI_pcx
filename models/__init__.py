import torch

from models.efficientnet import get_efficientnet_canonizer
from models.resnet import get_resnet18, get_resnet_canonizer
from models.vgg import get_vgg16, get_vgg_canonizer


MODELS = {
    "vgg16": get_vgg16,
    "resnet18": get_resnet18,
}

CANONIZERS = {
    "vgg16": get_vgg_canonizer,
    "efficientnet_b0": get_efficientnet_canonizer,
    "resnet18": get_resnet_canonizer,
}


def get_canonizer(model_name):
    assert model_name in list(CANONIZERS.keys()), f"No canonizer for model '{model_name}' available"
    return [CANONIZERS[model_name]()]


def get_fn_model_loader(model_name: str) -> torch.nn.Module:
    if model_name in MODELS:
        fn_model_loader = MODELS[model_name]
        return fn_model_loader
    else:
        raise KeyError(f"Model {model_name} not available")
