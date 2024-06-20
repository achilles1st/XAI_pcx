import os
import random
from argparse import ArgumentParser, Namespace

import h5py
import numpy as np
import torch

import torchvision

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
from sklearn.mixture import GaussianMixture
import zennit.image as zimage
from crp.image import imgify
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
from utils.helper import load_config, get_layer_names_model
from utils.lrp_composites import EpsilonPlusFlat
from utils.render import vis_opaque_img_border
torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

###########
CONFIG_FILE = "configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml"
LAYER = "features.28"
N_PROTOTYPES = 8
TOP_K_SAMPLE_PROTOTYPE = 6
CLASS_ID_ = 0
#########

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

args = Namespace(**load_config(CONFIG_FILE))
args.layer_name = LAYER

dataset_name = args.__dict__["dataset_name"]
model_name = args.__dict__["model_name"]
dataset = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split="train")
dataset_test = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split="test")
feature_path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"


for CLASS_ID in [CLASS_ID_]:
    print(f"### Class {CLASS_ID}")

    sample_ids = np.array(torch.load(f"{feature_path}/sample_ids_class_{CLASS_ID}_train.pt"))
    classes = np.array([dataset.get_target(i) for i in sample_ids])

    features_samples = torch.tensor(np.array(
        h5py.File(f"{feature_path}/eps_relevances_class_{CLASS_ID}_train.hdf5", "r")[args.layer_name]
    ))

    outputs = torch.cat(torch.load(f"{feature_path}/outputs_class_{CLASS_ID}_train.pt"))
    features_samples = features_samples[outputs.argmax(1) == CLASS_ID]
    sample_ids = sample_ids[outputs.argmax(1) == CLASS_ID]
    classes = classes[outputs.argmax(1) == CLASS_ID]
    # get all indices of samples of class CLASS_ID
    indices = sample_ids[classes == CLASS_ID]
    features = features_samples[classes == CLASS_ID]
    gmm = GaussianMixture(n_components=N_PROTOTYPES,
                        random_state=0,
                        covariance_type='full',
                        max_iter=10,
                        verbose=2,
                        reg_covar=1e-6,
                        n_init=1, init_params='kmeans').fit(features)

    distances = np.linalg.norm(features[:, None, :] - gmm.means_, axis=2)
    counts = np.unique(distances.argmin(1), return_counts=True)[1]
    counts_perc = counts / sum(counts) * 100
    prototype_samples = np.argsort(distances, axis=0)[:TOP_K_SAMPLE_PROTOTYPE]
    prototype_samples = indices[prototype_samples]
    # print(prototype_samples)

    ### PLOTTING CONCEPT MATRIX
    prototypes = torch.from_numpy(gmm.means_)
    top_concepts = torch.topk(prototypes.abs(), 3).indices.flatten().unique()
    top_concepts = top_concepts[prototypes[:, top_concepts].abs().amax(0).argsort(descending=True)]
    concept_matrix = prototypes[:, top_concepts].T
    N_CONCEPTS = len(top_concepts)

    model = get_fn_model_loader(model_name)(n_class=len(dataset.class_names), ckpt_path=args.ckpt_path).to("cuda")
    model.eval()
    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()

    layer_names = get_layer_names_model(model, model_name)
    layer_map = {layer: cc for layer in layer_names}

    print(layer_names)

    attribution = CondAttribution(model)

    fv_name = f"crp_files/{model_name}_{dataset_name}"
    fv = FeatureVisualization(attribution, dataset_test, layer_map, preprocess_fn=dataset.preprocessing,
                              path=fv_name, cache=None)


    topk_ind = [int(x) for x in top_concepts]

    ref_imgs = fv.get_max_reference(topk_ind, args.layer_name, "relevance", (0, 6), composite=composite, rf=True,
                                    plot_fn=vis_opaque_img_border)

    fig, axs = plt.subplots(nrows=N_CONCEPTS + 1, ncols=N_PROTOTYPES + 1, figsize=(N_PROTOTYPES + 6, N_CONCEPTS + 6), dpi=150,
                            gridspec_kw={'width_ratios': [6] + [1 for _ in range(N_PROTOTYPES)],
                                         'height_ratios': [6] + [1 for _ in range(N_CONCEPTS)]})
    for i in range(N_CONCEPTS):
        for j in range(N_PROTOTYPES):
            val = concept_matrix[i, j].item()
            axs[i + 1, j + 1].matshow(np.ones((1, 1)) * val if val >= 0 else np.ones((1, 1)) * val * -1,
                                      vmin=0,
                                      vmax=concept_matrix.abs().max(),
                                      cmap="Reds" if val >= 0 else "Blues")
            minmax = concept_matrix.abs().max() * 100 / 2
            cos = val * 100
            color = "white" if abs(cos) > minmax else "black"
            axs[i + 1, j + 1].text(0, 0, f"{cos:.1f}", ha="center", va="center", color=color, fontsize=15)
            axs[i + 1, j + 1].axis('off')
    resize = torchvision.transforms.Resize((120, 120))
    for i in range(N_PROTOTYPES):
        grid = make_grid(
            [resize(dataset[prototype_samples[j][i]][0])
             for j in range(6)],
            nrow=1,
            padding=0)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        img = imgify(grid)
        axs[0, i + 1].imshow(img)
        axs[0, i + 1].set_xticks([])
        axs[0, i + 1].set_yticks([])
        axs[0, i + 1].set_title(f"prototype {i} \ncovers {counts[i]} \n({counts_perc[i]:.0f}\%)")
        axs[0, 0].axis('off')


    for i in range(N_CONCEPTS):
        grid = make_grid(
            [resize(torch.from_numpy(np.asarray(i)).permute((2, 0, 1))) for i in ref_imgs[topk_ind[i]]],
            nrow=int(6 / 1),
            padding=0)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        axs[i + 1, 0].imshow(grid)
        axs[i + 1, 0].set_ylabel(f"concept {topk_ind[i]}")
        axs[i + 1, 0].set_yticks([])
        axs[i + 1, 0].set_xticks([])

    plt.tight_layout()
    os.makedirs(f"plot_files/{dataset_name}_{model_name}/prototype_data", exist_ok=True)
    plt.savefig(f"plot_files/{dataset_name}_{model_name}/prototype_data/{CLASS_ID}_prototypes_with_concepts_{LAYER}.pdf", dpi=300)

    plt.show()