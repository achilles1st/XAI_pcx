import os
from argparse import ArgumentParser

from typing import List


import torchvision
import yaml

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept

from crp.visualization import FeatureVisualization
from torchvision.transforms.functional import gaussian_blur

from torchvision.utils import make_grid


from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader
from utils.helper import get_layer_names_model
from utils.lrp_composites import EpsilonPlusFlat
from utils.render import vis_opaque_img_border
import torch
import numpy as np
import matplotlib.pyplot as plt
import zennit.image as zimage
from crp.image import imgify

def get_parser(fixed_arguments: List[str] = []):
    parser = ArgumentParser(
        description='Compute and display the top-k most relevant neurons for a given data sample/prediction.', )

    parser.add_argument('--sample_id', default=0)
    parser.add_argument('--layer_name', default="features.28")

    parser.add_argument('--config_file',
                        default="configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml")

    args = parser.parse_args()

    with open(parser.parse_args().config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["config_name"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    for k, v in config.items():
        if k not in fixed_arguments:
            setattr(args, k, v)

    return args


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

fv.dataset = train_dataset
data_p, target = fv.get_data_sample(sample_id, preprocessing=True)
fv.dataset = dataset

attr = attribution(data_p.requires_grad_(),
                   [{"y": target}],
                   composite,
                   record_layer=layer_names)
print("prediction:", attr.prediction.argmax(), "target:", target)
channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)

topk = torch.topk(channel_rels[0].abs(), n_concepts)
topk_ind = topk.indices.detach().cpu().numpy()
topk_rel = channel_rels[0, topk_ind]

print(topk_ind, topk_rel)
# conditional heatmaps
conditions = [{"y": target, layer_name: c} for c in topk_ind]
cond_attr = attribution(data_p.requires_grad_(), conditions, composite, record_layer=layer_names,)
cond_heatmap = cond_attr.heatmap

ref_imgs = fv.get_max_reference(topk_ind, layer_name, mode, (0, n_refimgs), composite=composite, rf=True,
                                plot_fn=vis_opaque_img_border, batch_size=12)

fig, axs = plt.subplots(n_concepts, 3, gridspec_kw={'width_ratios': [1, 1, n_refimgs / 4]},
                        figsize=(4 * n_refimgs / 4, 2.6 * n_concepts), dpi=200)
resize = torchvision.transforms.Resize((150, 150))

for r, row_axs in enumerate(axs):

    for c, ax in enumerate(row_axs):
        if c == 0:
            if r == 0:
                ax.set_title("input")
                fv.dataset = train_dataset
                img = imgify(fv.get_data_sample(sample_id, preprocessing=False)[0][0])
                fv.dataset = dataset
                ax.imshow(img)
            elif r == 1:
                ax.set_title("heatmap")
                img = imgify(attr.heatmap, cmap="bwr", symmetric=True)
                ax.imshow(img)
            else:
                ax.axis("off")

        if c == 1:
            if r == 0:
                ax.set_title("cond. heatmap")
            ax.imshow(imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=True))
            ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(topk_rel[r] * 100):2.1f}%")

        elif c >= 2:
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

        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
os.makedirs(f"plot_files/crp_topk/{fv_name}_{layer_name}", exist_ok=True)
plt.savefig(f"plot_files/crp_topk/{fv_name}_{layer_name}/sample_{sample_id}_wrt_{target}_crp_{mode}.png", dpi=300)
plt.savefig(f"plot_files/crp_topk/{fv_name}_{layer_name}/sample_{sample_id}_wrt_{target}_crp_{mode}.pdf", dpi=300)
plt.show()
