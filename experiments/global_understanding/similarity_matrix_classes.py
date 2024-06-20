import argparse

import random

import h5py
import numpy as np
import torch as torch
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from datasets import get_dataset
from models import get_fn_model_loader
from utils.helper import load_config

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

def get_args():
    parser = argparse.ArgumentParser(description='Outlier Detection')
    parser.add_argument('--config_file', type=str, default="configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml")
    parser.add_argument('--layer_name', type=str,
                        default="features.28")
    return parser.parse_args()


def main(model_name, dataset_name, data_path, layer_name):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"

    experiments = [
                    {"label": "CRV (LRP-eps)",
                    "type": "eps_relevances"},
                   ]

    for exp in experiments:
        exp['in_vecs_train'] = []
        exp['in_vecs_test'] = []
        exp['out_vecs_test'] = []
        exp['out_predicted_test'] = []

    in_sm_outputs = []


    print("Loading concept vectors...")
    classes_unique = []
    n_classes = 20

    for c in np.arange(0, n_classes):
        print("[IN] Loading class", c)
        for exp in experiments:
            data = torch.tensor(np.array(
                h5py.File(f"{path}/{exp['type']}_class_{c}_train.hdf5", "r")[layer_name]
            ))
            outputs = torch.cat(torch.load(f"{path}/outputs_class_{c}_train.pt"))
            exp['in_vecs_train'].append(data.float()[outputs.argmax(1) == c])
        classes_unique.append(c)

    print("Concept vectors loaded")

    for exp in experiments:
        exp['gmm'] = [GaussianMixture(n_components=1,
                                      reg_covar=1e-4,
                                      random_state=0).fit(v.numpy()) for v in exp['in_vecs_train']]
        exp['means'] = torch.cat([torch.from_numpy(gmm.means_) for gmm in exp['gmm']])

    print("GMMs fitted.")

    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)


    dataset = get_dataset(dataset_name)(data_path=data_path, preprocessing=True, split="test")
    class_names = dataset.class_names[:n_classes]
    class_names = [c.split(",")[0].replace("_", " ") for c in class_names]
    class_names = [c + "_" + str(i) for i, c in enumerate(class_names)]

    for exp in experiments:
        means = exp['means'].squeeze()
        cosine_similarity = torch.cosine_similarity(means[:, None], means[None], dim=-1)

        plt.figure(figsize=(5, 4), dpi=300)

        plt.imshow(cosine_similarity)

        cbar = plt.colorbar()
        cbar.ax.set_ylabel('similarity', rotation=270, labelpad=15)
        plt.xticks(np.arange(len(class_names)), rotation=45, ha="center", fontsize=8)
        # plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right", fontsize=6) #set fontsize of ticks
        plt.yticks(np.arange(len(class_names)), class_names)
        plt.tight_layout()
        plt.savefig(f"class_similarity_martrix_{dataset_name}_{model_name}.pdf", dpi=300)
        plt.show()


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    data_path = config.get('data_path', None)

    model = get_fn_model_loader(model_name)()

    main(model_name, dataset_name, data_path, args.layer_name)

