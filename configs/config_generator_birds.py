import copy
import os
import shutil

import yaml

config_dir = "configs/birds_configs"
shutil.rmtree(config_dir, onerror=lambda a, b, c: None)
os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("configs/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

_base_config = {
    'num_epochs': 100,
    'device': 'cuda',
    'dataset_name': 'birds',
    'loss': 'cross_entropy',
    'wandb_api_key': '',
    'wandb_project_name': '',
}

def store_local(config, config_name):
    config['ckpt_path'] = f"checkpoints/{config_name}/last.ckpt"
    config['batch_size'] = 32
    config['data_path'] = local_config['birds_dir']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name):
    config['ckpt_path'] = f"checkpoints/{config_name}/last.ckpt"
    config['batch_size'] = 64
    config['data_path'] = "/mnt/CUB_200_2011"

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name, layer_name in [
    ('vgg16', 'features.28'),
    ('resnet18', "last_conv"),
    ('efficientnet_b0', "last_conv")
]:

    _base_config['model_name'] = model_name

    for layer_name in [layer_name]:
        _base_config['layer_name'] = layer_name
        base_config = copy.deepcopy(_base_config)

        for lr in [
            5e-4 if model_name == "efficientnet_b0" else 1e-3,
        ]:
            base_config['lr'] = lr
            optim_name = "adam" if model_name == "efficientnet_b0" else "sgd"

            base_config['optimizer'] = optim_name

            for pretrained in [True,]:
                base_config['pretrained'] = pretrained
                ### VANILLA
                config_vanilla = copy.deepcopy(base_config)
                method = 'Vanilla'
                config_vanilla['method'] = method
                config_vanilla['lamb'] = 0.0
                config_name = f"{model_name}_birds_pretrained_{pretrained}_{method}_{optim_name}_lr{lr}_{layer_name}"
                store_local(config_vanilla, config_name)
                store_cluster(config_vanilla, config_name)