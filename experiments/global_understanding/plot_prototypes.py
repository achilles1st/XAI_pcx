import random
from argparse import Namespace
import os
import h5py
import numpy as np
import torch
import torchvision

from sklearn.mixture import GaussianMixture
import zennit.image as zimage
from crp.image import imgify
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets import get_dataset
from utils.helper import load_config

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

###########
CONFIG_FILE = "configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml"
N_PROTOTYPES = 8
TOP_K_SAMPLE_PROTOTYPE = 6
CLASS_ID_ = 0
#########

args = Namespace(**load_config(CONFIG_FILE))

dataset_name = args.__dict__["dataset_name"]
model_name = args.__dict__["model_name"]
dataset = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split="train")
feature_path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"

for CLASS_ID in [CLASS_ID_]:
    print(f"### Class {CLASS_ID}")

    sample_ids = np.array(torch.load(f"{feature_path}/sample_ids_class_{CLASS_ID}_train.pt"))
    classes = np.array([dataset.get_target(i) for i in sample_ids])

    features_samples = torch.tensor(np.array(
        h5py.File(f"{feature_path}/eps_relevances_class_{CLASS_ID}_train.hdf5", "r")[args.layer_name]
    ))

    indices = sample_ids[classes == CLASS_ID]
    features = features_samples[classes == CLASS_ID]

    outputs = torch.cat(torch.load(f"{feature_path}/outputs_class_{CLASS_ID}_train.pt"))
    features = features[outputs.argmax(1) == CLASS_ID]
    indices = indices[outputs.argmax(1) == CLASS_ID]
    classes = classes[outputs.argmax(1) == CLASS_ID]
  
    gmm = GaussianMixture(n_components=N_PROTOTYPES,
                        random_state=0,
                        covariance_type='full',
                        max_iter=10,
                        verbose=2,
                        n_init=1, init_params='kmeans').fit(features)
    distances = np.linalg.norm(features[:, None, :] - gmm.means_, axis=2)
    counts = np.unique(distances.argmin(1), return_counts=True)[1]
    counts_perc = counts / sum(counts) * 100
    prototype_samples = np.argsort(distances, axis=0)[:TOP_K_SAMPLE_PROTOTYPE]
    prototype_samples = indices[prototype_samples]

    # compute cosine similarity between gmm.means_
    mean_cosine_sim = torch.nn.functional.cosine_similarity(
        torch.from_numpy(gmm.means_),
        features.mean(0)[None])
    print(mean_cosine_sim)

    # plot prototypes in matplotlib in each row TOP_K_SAMPLE_PROTOTYPE
    fig, axs = plt.subplots(nrows=1, ncols=N_PROTOTYPES, figsize=(N_PROTOTYPES*1.2, 6*1.2), dpi=200)
    resize = torchvision.transforms.Resize((200, 200))
    for j in range(N_PROTOTYPES):
        grid = make_grid(
            [resize(dataset[prototype_samples[k][j]][0])
             for k in range(TOP_K_SAMPLE_PROTOTYPE)],
            nrow=1,
            padding=3)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        img = imgify(grid)
        axs[j].imshow(img)
        axs[j].set_xticks([])
        axs[j].set_yticks([])
        axs[j].set_title(f"Prototype {j} \ncovers {counts_perc[j]:.0f}\% \nsim. {mean_cosine_sim[j]:0.2f}")

    # save
    plt.tight_layout()
    os.makedirs(f"plot_files/{dataset_name}_{model_name}/prototype_data", exist_ok=True)
    plt.savefig(f"plot_files/{dataset_name}_{model_name}/prototype_data/{mean_cosine_sim.min():0.2f}_{CLASS_ID}_prototypes.pdf", dpi=300)

    plt.show()