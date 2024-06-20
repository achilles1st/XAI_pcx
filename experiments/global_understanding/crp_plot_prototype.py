import os
from argparse import ArgumentParser

import h5py
import torch
import torch.nn as nn
import torchvision
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import *
from crp.visualization import FeatureVisualization
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from torchvision.utils import make_grid
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader
from utils.helper import load_config, get_layer_names_model
from utils.render import vis_opaque_img_border


def get_parser(fixed_arguments: List[str] = []):
    parser = ArgumentParser(
        description='Compute and display the top-k most relevant neurons for a given data sample/prediction.', )

    parser.add_argument('--class_id', default=0)
    parser.add_argument('--layer_name', default="features.28")
    parser.add_argument('--num_prototypes', default=8)
    parser.add_argument('--prototype', default=5)

    parser.add_argument('--config_file',
                        default="configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml")

    args = parser.parse_args()

    config = load_config(args.config_file)

    for k, v in config.items():
        if k not in fixed_arguments:
            setattr(args, k, v)

    return args


args = get_parser(["layer_name"])

model_name = args.model_name
dataset_name = args.dataset_name

fv_name = f"crp_files/{model_name}_{dataset_name}"

batch_size = 50
layer_name = args.layer_name
n_concepts = 5
n_refimgs = 12
mode = "relevance"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split="test")


ckpt_path = args.ckpt_path

model = get_fn_model_loader(model_name)(n_class=len(dataset.class_names), ckpt_path=ckpt_path).to(device)
model.eval()

canonizers = get_canonizer(model_name)
composite = EpsilonPlusFlat(canonizers)
cc = ChannelConcept()

layer_names = get_layer_names_model(model, model_name)
layer_map = {layer: cc for layer in layer_names}

attribution = CondAttribution(model)

fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                          path=fv_name, cache=None)

if not os.listdir(fv.RelMax.PATH):
    fv.run(composite, 0, len(dataset), batch_size=batch_size)


feature_path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"

features = torch.tensor(
                h5py.File(f"{feature_path}/eps_relevances_class_{args.class_id}_train.hdf5", "r")[args.layer_name]
            )

outputs = torch.cat(torch.load(f"{feature_path}/outputs_class_{args.class_id}_train.pt"))
features = features[outputs.argmax(dim=1) == args.class_id]
samples = np.array(torch.load(f"{feature_path}/sample_ids_class_{args.class_id}_train.pt"))
samples = samples[outputs.argmax(dim=1) == args.class_id]

gmm = GaussianMixture(n_components=args.num_prototypes,
                      random_state=0,
                      covariance_type='full',
                      max_iter=10,
                      verbose=2,
                      n_init=1,).fit(features)

mean = gmm.means_[args.prototype]
means = torch.tensor(gmm.means_)

closest_sample_to_mean = ((features - mean[None])).pow(2).sum(dim=1).argsort()[0].item()
# print(samples[((features - mean[None])).pow(2).sum(dim=1).argsort()[:5]])
mean = torch.from_numpy(mean)
max_rel = [features[:, i].argmax().item() for i in torch.topk(mean, n_concepts).indices]

dataset_train = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=True, split="train")

samples_of_class = samples
closest_sample_to_mean = samples_of_class[closest_sample_to_mean]

data_p, target = dataset_train[closest_sample_to_mean]
data_p = data_p.to(device)[None]
# print([samples_of_class[i] for i in max_rel])
# print(closest_sample_to_mean)
data_topk = [dataset_train[samples_of_class[i]][0].to(device)[None] for i in max_rel]

attr = attribution(data_p.requires_grad_(),
                   [{"y": target}],
                   composite,
                   record_layer=[layer_name])
print("prediction:", attr.prediction.argmax(), "target:", target)
channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)

topk = torch.topk(mean, n_concepts)
topk_ind = topk.indices.detach().cpu().numpy()
topk_rel = topk.values.detach().cpu().numpy()

print(topk_ind, topk_rel)

# conditional heatmaps
conditions = [{"y": target, layer_name: c} for c in topk_ind]
cond_heatmap_p, _, _, _ = attribution(data_p.requires_grad_(), conditions, composite)

cond_heatmap, _, _, _ = attribution(torch.cat(data_topk, 0).requires_grad_(), conditions, composite)

ref_imgs = fv.get_max_reference(topk_ind, layer_name, mode, (0, n_refimgs), composite=composite, rf=True,
                                plot_fn=vis_opaque_img_border)

fig, axs = plt.subplots(n_concepts, 5, gridspec_kw={'width_ratios': [1, 1, 1, 1, n_refimgs / 4]},
                        figsize=(4 * n_refimgs / 4 + 5, 2.6 * n_concepts), dpi=200)
resize = torchvision.transforms.Resize((150, 150))

for r, row_axs in enumerate(axs):

    for c, ax in enumerate(row_axs):
        if c == 0:
            if r == 0:
                ax.set_title("prototype")
                img = imgify(dataset_train.reverse_augmentation(data_p[0]))
                ax.imshow(img)
            elif r == 1:
                ax.set_title("heatmap")
                img = imgify(attr.heatmap, cmap="bwr", symmetric=True)
                ax.imshow(img)
            else:
                ax.axis("off")
        if c == 2:
            ax.set_title("concept-reference" if r == 0 else None)
            ax.imshow(imgify(dataset_train.reverse_augmentation(data_topk[r][0])))

        if c == 1:
            if r == 0:
                ax.set_title("cond. heatmap")
            ax.imshow(imgify(cond_heatmap_p[r], symmetric=True, cmap="bwr", padding=True))
            ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(topk_rel[r] * 100):2.1f}%")
        if c == 3:
            if r == 0:
                ax.set_title("cond. heatmap (ref)")
            ax.imshow(imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=True))

        elif c == 4:
            if r == 0:
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
plt.savefig(f"plot_files/crp_topk/{fv_name}_{layer_name}/sample_{closest_sample_to_mean}_wrt_{target}_crp_{mode}.png", dpi=300)

plt.show()
