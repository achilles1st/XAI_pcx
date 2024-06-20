import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets import get_dataset
from model_training.base_pl import Vanilla
from models import get_fn_model_loader

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_parser():
    parser = ArgumentParser(
        description='Train models.', )
    parser.add_argument('--config_file',
                        default="configs/config.yaml")
    parser.add_argument('--num_gpu', default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)
    config["config_file"] = config_file
    config_name = os.path.basename(config_file)[:-5]
    start_model_correction(config, config_name, args.num_gpu)


def start_model_correction(config, config_name, num_gpu):
    """ Starts model correction for given config file.

    Args:
        config (dict): Dictionary with config parameters for training.
        config_name (str): Name of given config
        num_gpu (int): Number of GPUs
    """

    # Initialize WandB
    if config["wandb_project_name"]:
        wandb_api_key = config["wandb_api_key"]
        wandb_project_name = config["wandb_project_name"]
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb_id = f"{config_name}" if config.get('unique_wandb_ids', True) else None
        logger_ = WandbLogger(project=wandb_project_name, name=f"{config_name}", id=wandb_id, config=config)

    # Load Dataset
    dataset_name = config["dataset_name"]
    data_path = config["data_path"]
    dataset = get_dataset(dataset_name)
    dataset_train = dataset(data_path=data_path, preprocessing=True, split="train")
    dataset_val = dataset(data_path=data_path, preprocessing=True, split="val")

    # Load Model
    model_name = config["model_name"]
    fn_model_loader = get_fn_model_loader(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fn_model_loader(
        pretrained=config["pretrained"],
        n_class=len(dataset_train.class_names)).to(device)

    model = Vanilla(model, config)

    # Define Optimizer and Loss function
    optimizer_name = config["optimizer"]
    lr = config["lr"]

    model.set_optimizer(optimizer_name, model.parameters(), lr)
    model.set_loss("cross_entropy", weights=getattr(dataset_train, "weights", None))

    print(f"Number of samples: {len(dataset_train)} (train) / {len(dataset_val)} (val)")

    batch_size = config["batch_size"]
    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    checkpoint_callback = ModelCheckpoint(monitor="valid_acc",
                                          dirpath=f"checkpoints/{config_name}",
                                          filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
                                          auto_insert_metric_name=False,
                                          save_last=True,
                                          save_weights_only=True,
                                          mode="max")

    trainer = Trainer(callbacks=[
        checkpoint_callback,
    ],
        devices=num_gpu,
        max_epochs=config.get("num_epochs", 100),
        accelerator="gpu",
        logger=logger_)

    trainer.fit(model, dl_train, dl_val)


if __name__ == "__main__":
    main()
