import argparse
import os
import random

import h5py
import numpy as np
import torch
import wandb

from sklearn.mixture import GaussianMixture

from datasets import get_dataset
from models import get_fn_model_loader

from utils.helper import load_config, get_layer_names_model

random.seed(10)
torch.random.manual_seed(10)
np.random.seed(10)



def get_args():
    parser = argparse.ArgumentParser(description='Measure Sparseness')
    parser.add_argument('--config_file', type=str, default="configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml")
    parser.add_argument('--layer_name', type=str, default=None)
    parser.add_argument('--num_prototypes', type=int, default=1)
    return parser.parse_args()


def main(model_name, dataset_name, layer_name, num_prototypes):

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

    if num_prototypes > 1:
        experiments = experiments[1:2]

    for exp in experiments:
        exp['vecs'] = []
        exp['vecs_train'] = []
        exp['samples'] = []

    print("Loading concept vectors...")

    path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"

    n_classes = {
        "cifar10" : 10,
        "imagenet": 20,
        "birds": 200,
    }[dataset_name]

    classes_unique = []
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
            exp['vecs'].append(data.float())
            exp['samples'].append(torch.load(f"{path}/sample_ids_class_{c}_test.pt"))
            if c not in classes_unique:
                classes_unique.append(c)


    for exp in experiments:
        exp['means'] = torch.stack([v.mean(0) for v in exp['vecs_train']], 0)
        print("Fitting Gaussians...")
        if num_prototypes == 0:
            continue
        exp['gmm'] = [GaussianMixture(n_components=num_prototypes,
                                      covariance_type='full' if exp['means'][0].shape[-1] < 1000 else 'diag',
                                      max_iter=100,
                                      random_state=0,
                                      reg_covar=1e-4 if model_name == "efficientnet_b0" else 1e-6,
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

    print("Concept vectors loaded")

    for exp in experiments:
        if "gmms_individual" in exp.keys():
            print("Computing Sparseness...")
            sparseness = []
            for i, g in enumerate(exp['gmms_individual']):
                vecs = torch.tensor(np.array(
                    [gmm.means_ for gmm in g]
                ))
                uniform = torch.ones_like(vecs[0:1]) / vecs.shape[-1]
                sparseness.append(1 - torch.nn.functional.cosine_similarity(vecs.abs(), uniform, dim=-1))
            exp['sparseness'] = torch.cat(sparseness, 0).mean()
            exp['sparseness_se'] = torch.cat(sparseness, 0).std() / np.sqrt(len(torch.cat(sparseness, 0)))
        else:
            vecs = exp["means"]
            uniform = torch.ones_like(vecs[0:1]) / vecs.shape[-1]
            sparseness = 1 - torch.nn.functional.cosine_similarity(vecs.abs(), uniform, dim=-1)
            exp['sparseness'] = sparseness.mean()
            exp['sparseness_se'] = sparseness.std() / np.sqrt(len(sparseness))

        wandb.log({f"sparseness_{layer_name}_{exp['label']}": exp['sparseness']})
        wandb.log({f"sparseness_{layer_name}_{exp['label']}_se": exp['sparseness_se']})
        print(f"Sparseness {layer_name} {exp['label']}: {exp['sparseness']}")




if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    layer_name = args.layer_name
    batch_size = 512 if "imagenet" not in dataset_name else 16
    data_path = config.get('data_path', None)
    num_prototypes = config.get("num_prototypes", args.num_prototypes)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'] if config.get("unique_wandb_ids") else None,
                   project=config['wandb_project_name'], config=config,  name=config['wandb_id'])

    if layer_name is not None:
        main(model_name, dataset_name, layer_name, num_prototypes)
    else:
        dataset = get_dataset(dataset_name)(data_path=data_path, preprocessing=False, split="test")
        n_classes = dataset.num_classes
        model = get_fn_model_loader(model_name)(n_class=len(dataset.class_names), ckpt_path=None)
        layer_names = get_layer_names_model(model, model_name)

        for layer_name in layer_names:
            main(model_name, dataset_name, layer_name, num_prototypes)



