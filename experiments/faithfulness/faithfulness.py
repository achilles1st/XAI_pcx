import argparse
import os
import random

import h5py
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from zennit.core import Composite

from datasets import get_dataset
from models import CANONIZERS, get_fn_model_loader

from utils.helper import load_config, get_layer_names_model

random.seed(10)
torch.random.manual_seed(10)
np.random.seed(10)



def get_args():
    parser = argparse.ArgumentParser(description='Measure Faithfulness')
    parser.add_argument('--config_file', type=str, default="configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml")
    parser.add_argument('--layer_name', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=300)
    parser.add_argument('--num_prototypes', type=int, default=1)
    parser.add_argument('--channel_fraction', type=float, default=1.0)
    return parser.parse_args()


def main(model_name, dataset_name, data_path, num_samples, layer_name, batch_size, num_prototypes, channel_fraction, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = get_dataset(dataset_name)(data_path=data_path, preprocessing=False, split="test")
    model = get_fn_model_loader(model_name)(n_class=len(dataset.class_names), ckpt_path=ckpt_path).to(device)
    model_masked = get_fn_model_loader(model_name)(n_class=len(dataset.class_names), ckpt_path=ckpt_path).to(device)
    model = model.to(device)
    model_masked = model_masked.to(device)
    model.eval()
    model_masked.eval()

    experiments = [
        {"label": "CRV (LRP-zplus)",
         "type": "zplus_relevances"},
        {"label": "CRV (LRP-eps)",
         "type": "eps_relevances"},
        {"label": "CRV (I*G)",
         "type": "ig_relevances"},
        {"label": "CRV (GuidedBP)",
         "type": "gbp_relevances"},
        {"label": "CAV (max)",
         "type": "max_activations"},
        {"label": "CAV (mean)",
         "type": "mean_activations"},
                   ]

    for exp in experiments:
        exp['vecs'] = []
        exp['vecs_train'] = []
        exp['samples'] = []

    print("Loading concept vectors...")

    path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"

    classes_unique = []

    n_classes = {
        "cifar10" : 10,
        "imagenet": 20,
        "birds": 200,
    }[dataset_name]

    for c in np.arange(0, n_classes):
        for exp in experiments:
            data = torch.tensor(np.array(
                h5py.File(f"{path}/{exp['type']}_class_{c}_train.hdf5", "r")[layer_name]
            ))
            outputs = torch.cat(torch.load(f"{path}/outputs_class_{c}_train.pt"))
            exp['vecs_train'].append(data.float()[outputs.argmax(1) == c])
            data = torch.tensor(np.array(
                h5py.File(f"{path}/{exp['type']}_class_{c}_test.hdf5", "r")[layer_name]
            ))
            outputs = torch.cat(torch.load(f"{path}/outputs_class_{c}_test.pt"))
            exp['vecs'].append(data.float()[outputs.argmax(1) == c])
            exp['samples'].append(list(np.array(torch.load(f"{path}/sample_ids_class_{c}_test.pt"))[outputs.argmax(1) == c]))
            if c not in classes_unique:
                classes_unique.append(c)

    for exp in experiments:
        exp['means'] = torch.stack([v.mean(0) for v in exp['vecs_train']], 0)
        print("Fitting Gaussians...")
        if num_prototypes == 0:
            continue
        exp['gmm'] = [GaussianMixture(n_components=num_prototypes,
                                      covariance_type='full' if exp['means'][0].shape[-1] < 1000 else 'diag',
                                      reg_covar=1e-4,
                                      max_iter=100,
                                      random_state=0,
                                      n_init=1,).fit(v.numpy()) for v in exp['vecs_train']]
        exp['gmms_individual'] = [
            [GaussianMixture(n_components=1,
                             covariance_type='full' if exp['means'][0].shape[-1] < 1000 else 'diag') for p in
             range(num_prototypes)] for g in exp['gmm']
        ]
        [
            [exp['gmms_individual'][i][p]._set_parameters([param[p:p + 1] if j > 0 else param[p:p + 1] * 0 + 1
                                                           for j, param in enumerate(g._get_parameters())]) for p in
             range(num_prototypes)] for i, g in enumerate(exp['gmm'])
        ]


    # adding random experiment
    experiments.append({"label": "random",
                        "type": "random",
                        "samples": experiments[0]['samples'],
                        "means": torch.rand_like(experiments[0]['means'])})

    N = int(exp['means'].shape[-1] * 1)
    hooks = []
    neuron_indices = []

    composite = Composite(canonizers=[CANONIZERS[model_name]()])

    plt.figure(dpi=300)
    total_steps = int(N * channel_fraction)
    n_steps = min(30, total_steps)
    steps = np.round(np.linspace(0, total_steps, n_steps)).astype(int)

    with composite.context(model_masked) as model_masked_mod:

        def hook(m, i, o):
            for b, batch_indices in enumerate(neuron_indices):
                o[b][batch_indices] = o[b][batch_indices] * 0

        for n, m in model_masked_mod.named_modules():
            if n == layer_name:
                hooks.append(m.register_forward_hook(hook))

        class1 = np.array([random.choice(classes_unique) for _ in range(num_samples)])
        samples = [experiments[0]["samples"][classes_unique.index(c)][list(np.where(class1 == c)[0]).index(j)] for j, c in
                   enumerate(class1)]
        all_samples = np.array(samples)
        batches = int(np.ceil(len(all_samples) / batch_size))
        diffs = {}
        for i, exp in enumerate(tqdm(experiments)):
            if "gmms_individual" in exp.keys():
                gmms = [exp["gmms_individual"][classes_unique.index(c)] for c in class1]
                features = [(exp["vecs"][classes_unique.index(c)])[exp["samples"][classes_unique.index(c)].index(idx)] for idx, c in zip(samples, class1)]
                class1_means = torch.stack(
                    [torch.tensor(gmm_[np.argmax([g.score_samples(f[None]) for g in gmm_])].means_[0])
                                            for gmm_, f in zip(gmms, features)])
            elif exp["type"] == "random":
                class1_means = torch.rand_like(class1_means)
            elif num_prototypes == 0:
                features = [(exp["vecs"][classes_unique.index(c)])[exp["samples"][classes_unique.index(c)].index(idx)]
                            for idx, c in zip(samples, class1)]
                class1_means = torch.stack(features)


            diffs_ = []
            topk_concepts = torch.topk(class1_means, N)
            for k in steps:
                diff = []
                neuron_indices_all = topk_concepts.indices[:, :k]

                for b in range(batches):
                    neuron_indices = neuron_indices_all[b * batch_size: (b + 1) * batch_size]
                    data = torch.stack([dataset[s][0] for s in all_samples[b * batch_size: (b + 1) * batch_size]])
                    data = data.to(device)
                    masked_out = model_masked(data).detach().cpu()
                    out = model(data).detach().cpu()
                    out_diff = masked_out - out
                    diff.extend(list(torch.gather(out_diff.t(),0,
                                torch.tensor(class1)[None, b * batch_size: ( b + 1) * batch_size]).numpy()[0, :]))
                diffs_.append(diff)
            diffs_ = np.array(diffs_)

            AUC = [np.trapz(diffs_[:, i::8].mean(1), steps / N) for i in range(8)]
            AUC_err = np.std(AUC) / np.sqrt(len(AUC))
            AUC = np.mean(AUC)
            diffs_ = diffs_.mean(1)
            plt.plot(steps, diffs_, 'o--', label=exp["label"] + f"({AUC:.3f})")
            diffs[exp["label"]] = diffs_

            wandb.log({f"pert_AUC_{layer_name}_{exp['label']}": AUC})
            wandb.log({f"pert_AUC_{layer_name}_{exp['label']}_err": AUC_err})

    plt.legend()
    plt.xlabel("removed concepts")
    plt.ylabel("mean logit change")
    path = f"results/faithfulness/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "/data", exist_ok=True)
    os.makedirs(path + "/figs", exist_ok=True)

    plt.savefig(f"{path}/figs/class_perturbation_{layer_name}.pdf", dpi=300, transparent=True)
    torch.save(diffs, f"{path}/data/class_perturbation_{layer_name}.pth")
    plt.show()


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    layer_name = args.layer_name
    batch_size = 512 if "cifar" in dataset_name else 12
    data_path = config.get('data_path', None)
    ckpt_path = config.get('ckpt_path', None)

    num_prototypes = config.get("num_prototypes", args.num_prototypes)
    config["channel_fraction"] = args.channel_fraction

    if config.get('wandb_api_key', None) and False:
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'] if config.get("unique_wandb_ids") else None,
                   project=config['wandb_project_name'], config=config,  name=config['wandb_id'])

    if layer_name is not None:
        main(model_name, dataset_name, data_path, args.num_samples, layer_name, batch_size, num_prototypes, args.channel_fraction, ckpt_path)
    else:
        dataset = get_dataset(dataset_name)(data_path=data_path, preprocessing=False, split="test")
        n_classes = dataset.num_classes
        model = get_fn_model_loader(model_name)(n_class=len(dataset.class_names), ckpt_path=ckpt_path)
        layer_names = get_layer_names_model(model, model_name)

        for layer_name in layer_names:
            main(model_name, dataset_name, data_path, args.num_samples, layer_name, batch_size, num_prototypes, args.channel_fraction, ckpt_path)



