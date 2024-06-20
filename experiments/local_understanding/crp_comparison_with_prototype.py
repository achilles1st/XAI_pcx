import os
from argparse import ArgumentParser
from typing import List

import h5py
import torch.nn as nn
import torchvision
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names

from crp.visualization import FeatureVisualization
from sklearn.mixture import GaussianMixture

from torchvision.utils import make_grid
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader
from utils.helper import load_config, get_layer_names_model
from utils.render import vis_opaque_img_border
import torch
import numpy as np
import matplotlib.pyplot as plt
import zennit.image as zimage
from crp.image import imgify


# EXAMPLES: 12064 for ImageNet VGG (ostrich)
# EXAMPLES: 29457 for ImageNet ResNet (eagle)
# EXAMPLES: 522589 for ImageNet VGG (ambulance)
# EXAMPLES: 360929 for ImageNet ResNet (tiger cat)
def get_parser(fixed_arguments: List[str] = []):
    parser = ArgumentParser(
        description='Compute and display the top-k most relevant neurons for a given data sample/prediction.', )

    parser.add_argument('--sample_id', default=None)  # If none, automatically take outlier of class class_id
    parser.add_argument('--class_id', default=0)
    parser.add_argument('--layer_name', default="features.28")
    parser.add_argument('--split', default="train")  # test not implemented
    parser.add_argument('--num_prototypes', default=8)
    parser.add_argument('--config_file',
                        default="configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml"
                        )

    args = parser.parse_args()

    config = load_config(args.config_file)

    for k, v in config.items():
        if k not in fixed_arguments:
            setattr(args, k, v)

    return args

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

args = get_parser(["layer_name"])

model_name = args.model_name
dataset_name = args.dataset_name

fv_name = f"crp_files/{model_name}_{dataset_name}"
sample_id = args.sample_id
batch_size = 50
layer_name = args.layer_name
n_concepts = 5
n_refimgs = 12
mode = "relevance"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split="test")
train_dataset = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split="train")

ckpt_path = args.ckpt_path

model = get_fn_model_loader(model_name)(n_class=len(dataset.class_names), ckpt_path=ckpt_path).to(device)
model.eval()
# print(dataset.get_sample_name(562))
canonizers = get_canonizer(model_name)
composite = EpsilonPlusFlat(canonizers)
cc = ChannelConcept()

layer_names = get_layer_names_model(model, model_name)
layer_map = {layer: cc for layer in layer_names}

print(layer_names)

attribution = CondAttribution(model)

fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                          path=fv_name, cache=None)

if not os.listdir(fv.RelMax.PATH):
    fv.run(composite, 0, len(dataset), batch_size=batch_size)

if args.sample_id:
    fv.dataset = train_dataset if args.split == "train" else dataset
    data, target = fv.get_data_sample(sample_id, preprocessing=True)
    fv.dataset = dataset
    print(target)
    target = args.class_id
else:
    target = args.class_id

### Get Prototype
feature_path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"

features = torch.tensor(
                h5py.File(f"{feature_path}/eps_relevances_class_{target}_train.hdf5", "r")[args.layer_name]
            )

threshold = 0
outputs = torch.cat(torch.load(f"{feature_path}/outputs_class_{target}_train.pt"))
features = features[outputs.argmax(dim=1) == args.class_id]
# features = features[(outputs[:, args.class_id] >= threshold) & (outputs.argmax(dim=1) == args.class_id)]
samples = np.array(torch.load(f"{feature_path}/sample_ids_class_{target}_train.pt"))
samples = samples[outputs.argmax(dim=1) == args.class_id]
# samples = samples[(outputs[:, args.class_id] >= threshold) & (outputs.argmax(dim=1) == args.class_id)]

num_prototypes = args.num_prototypes
gmm = GaussianMixture(n_components=num_prototypes,
                      # covariance_type='diag',
                      reg_covar=1e-5,
                      random_state=0).fit(features)

prototype_gmms = [GaussianMixture(n_components=1, covariance_type='full',) for p in range(num_prototypes)]
for p, g_ in enumerate(prototype_gmms):
    g_._set_parameters([
        param[p:p + 1] if j > 0 else param[p:p + 1] * 0 + 1
        for j, param in enumerate(gmm._get_parameters())
    ])

scores = gmm.score_samples(features)

if args.sample_id is None:
    print("CHOOSING OUTLIER SAMPLE AUTOMATICALLY")
    argsort = np.argsort(scores)
    samples_sorted = samples[argsort]
    sample_id = samples_sorted[0]
    print("sample_id", sample_id)
    fv.dataset = train_dataset
    data, target = fv.get_data_sample(sample_id, preprocessing=True)
    fv.dataset = dataset


# Compute latent relevances
attr = attribution(data.requires_grad_(),[{"y": target}], composite, record_layer=[layer_name])
print("prediction:", attr.prediction.argmax(), "target:", target)
channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)
score_sample = gmm.score_samples(channel_rels.detach().cpu())
# Get closest prototype
likelihoods = [g_.score_samples(channel_rels.detach().cpu()) for g_ in prototype_gmms]

mean = gmm.means_[np.argmax(likelihoods)]
mean = torch.from_numpy(mean)
closest_sample_to_mean = ((features - mean[None])).pow(2).sum(dim=1).argmin().item()

dataset_train = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=True, split="train")

samples_of_class = samples
closest_sample_to_mean = samples_of_class[closest_sample_to_mean]

data_p, target = dataset_train[closest_sample_to_mean]
data_p = data_p.to(device)[None]

# Deciding concepts to show

topk = torch.topk(channel_rels[0].abs(), n_concepts)
topk_proto = torch.topk(mean.abs(), n_concepts)
topk_ind = topk.indices.detach().cpu().numpy()
topk_ind_proto = topk_proto.indices.detach().cpu().numpy()
top_ind = [topk_ind_proto[i//2] if i%2 == 0 else topk_ind[i//2] for i in np.arange(n_concepts*2)]
indexes = np.unique(top_ind, return_index=True)[1]
topk_ind = [top_ind[index] for index in sorted(indexes)][:3]

topk = torch.topk(torch.stack([channel_rels[0].to(mean), mean], -1).abs().amax(-1), n_concepts).indices.detach().cpu().numpy()
topk_ind = topk[:3].tolist()

diff_ = torch.topk((channel_rels[0].detach().cpu() - mean).abs(), 5).indices
diff_ = [ind for ind in diff_ if ind not in topk_ind]
topk_ind = topk_ind + diff_[:2]


topk_rel = channel_rels[0, topk_ind]

print(topk_ind, topk_rel)
# conditional heatmaps
conditions = [{"y": target, layer_name: c} for c in topk_ind]
attr_p = attribution(data_p.requires_grad_(),[{"y": target}], composite, record_layer=[layer_name])
cond_heatmap_p, _, _, _ = attribution(data_p.requires_grad_(), conditions, composite)
cond_heatmap, _, _, _ = attribution(data.requires_grad_(), conditions, composite)
ref_imgs = fv.get_max_reference(topk_ind, layer_name, mode, (0, n_refimgs), composite=composite, rf=True,
                                plot_fn=vis_opaque_img_border)


fig, axs = plt.subplots(n_concepts, 5, gridspec_kw={'width_ratios': [1, 1, n_refimgs / 4, 1, 1]},
                        figsize=(4 * n_refimgs / 4, 1.8 * n_concepts), dpi=200)
resize = torchvision.transforms.Resize((150, 150))

for r, row_axs in enumerate(axs):

    for c, ax in enumerate(row_axs):
        if c == 0:
            if r == 0:
                ax.set_title("input")
                fv.dataset = train_dataset if args.split == "train" else dataset
                img = imgify(fv.get_data_sample(sample_id, preprocessing=False)[0][0])
                fv.dataset = dataset
                ax.imshow(img)
            elif r == 1:
                ax.set_title("heatmap")
                img = imgify(attr.heatmap, cmap="bwr", symmetric=True)
                ax.imshow(img)
            elif r == 2:
                ax.set_title("class likelihood")
                a = ax.hist(scores, bins=20, color='k')
                ax.vlines(score_sample, 0, a[0].max(), linestyle='--', linewidth=3, label="sample")
                ax.legend()
                ax.set_ylabel("density")
                ax.set_xlabel("log-likelihood")
                ax.set_yticks([])
                ax.set_xticks([])
            else:
                ax.axis("off")

        if c == 1:
            if r == 0:
                ax.set_title("localization")
            ax.imshow(imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=True))
            ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(channel_rels[0][topk_ind[r]] * 100):2.1f}\%")

            delta_R = (channel_rels[0][topk_ind[r]].round(decimals=3) - mean[topk_ind[r]].round(decimals=3)) * 100
            textstr = f"$\\Delta R = {'+' if delta_R>0 else ''}{delta_R:2.1f}\%$"

            if abs(delta_R) > 3:
                c = "#f57a5eb3"
            elif abs(delta_R) > 1.5:
                c = "#f6b951b3"
            else:
                c = "#91d068b3"
            props = dict(boxstyle='round', facecolor=c, alpha=1.0, edgecolor="none")
            # place a text box in upper left in axes coords
            ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props, ha="center")

        elif c == 2:
            if r == 0 and c == 2:
                ax.set_title("concept visualization")
            grid = make_grid(
                [resize(torch.from_numpy(np.asarray(i)).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]],
                nrow=int(n_refimgs / 2),
                padding=0)
            grid = np.array(zimage.imgify(grid.detach().cpu()))
            img = imgify(ref_imgs[topk_ind[r]][c - 2], padding=True)
            ax.imshow(grid)
            ax.yaxis.set_label_position("right")
        elif c == 4:
            if r == 0:
                ax.set_title("prototype")
                fv.dataset = train_dataset
                img = imgify(fv.get_data_sample(closest_sample_to_mean, preprocessing=False)[0][0])
                fv.dataset = dataset
                ax.imshow(img)
            elif r == 1:
                ax.set_title("heatmap")
                img = imgify(attr_p.heatmap, cmap="bwr", symmetric=True)
                ax.imshow(img)
            else:
                ax.axis("off")
        if c == 3:
            if r == 0:
                ax.set_title("localization")
            ax.imshow(imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", padding=True))
            ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(mean[topk_ind[r]] * 100):2.1f}\%")

        ax.set_xticks([])
        ax.set_yticks([])

# add horizontal line
ax.plot([1/6, 5/6], [2/5 - 0.01, 2/5 - 0.01], color='lightgray', lw=1.5, ls="--",
        transform=plt.gcf().transFigure, clip_on=False)
ax.text(5/6, 2/5 - 0.008, "concepts sorted by $|R|$", transform=plt.gcf().transFigure, fontsize=10,
        verticalalignment='bottom', ha="right", clip_on=False, in_layout=False, color="gray")
ax.text(5/6, 2/5 - 0.013, "remaining concepts sorted by $|\\Delta R|$", transform=plt.gcf().transFigure, fontsize=10,
        verticalalignment='top', ha="right", clip_on=False, in_layout=False, color="gray")
plt.tight_layout()

plt.savefig(f"plot_files/{dataset_name}_{model_name}/prototype_data/{target}_{sample_id}_prototype_comparison.pdf", dpi=300)

plt.show()
