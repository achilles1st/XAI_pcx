import argparse
import os
import random

import h5py
import numpy as np
import torch as torch
import wandb
from crp.helper import get_layer_names
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from datasets.imagenet import group_class_ids, groups, class_groups
from experiments.coverage.identify_subclusters import solve_hungarian
from experiments.stability.km_matcher import KMMatcher
from models import get_fn_model_loader
from utils.helper import load_config

np.random.seed(1)
torch.manual_seed(0)
random.seed(0)

def get_args():
    parser = argparse.ArgumentParser(description='Measure Coverage')
    parser.add_argument('--config_file', type=str,
                        default="configs/coverage/local/vgg16_imagenet_num_prototypes_8.yaml"
                        )
    return parser.parse_args()



def main(model_name, dataset_name, layer_name, feature_type, group, num_clusters_list):

    path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"

    classes_to_combine = [
        *groups[group]
    ]
    # pick a subset of classes randomly
    N = 8
    classes_to_combine = random.sample(classes_to_combine, N)

    print(f"### Classes: {classes_to_combine}")
    print(f"### Layer: {layer_name}")
    print(f"### Feature type: {feature_type}")


    Nth = 2
    features = [torch.from_numpy(np.array(
        h5py.File(f"{path}/{feature_type}_class_{CLASS_ID}_train.hdf5", "r")[layer_name][::Nth]
    )) for CLASS_ID in classes_to_combine]
    sub_classes = [id_ * torch.ones_like(f[:, 0]) for id_, f in enumerate(features)]

    features_test = [torch.from_numpy(np.array(
        h5py.File(f"{path}/{feature_type}_class_{CLASS_ID}_train.hdf5", "r")[layer_name][1::Nth]
    )) for CLASS_ID in classes_to_combine]
    sub_classes_test = [id_ * torch.ones_like(f[:, 0]) for id_, f in enumerate(features_test)]

    features = torch.cat(features, dim=0)
    features_test = torch.cat(features_test, dim=0)
    sub_classes_test = torch.cat(sub_classes_test, dim=0)

    vals = {
        "kmeans": [],
        "gmm_euc": [],
        "gmm": [],
    }

    for num_clusters in num_clusters_list:

        k_means = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(features)


        def gmm_bic_score(estimator, X):
            return -estimator.bic(X)

        param_grid = {
            "reg_covar": [
                  1e1,
                  1e-3,
                  1e-6
                          ],
        }
        grid_search = GridSearchCV(
            GaussianMixture(n_components=num_clusters, random_state=0,
                            max_iter=10,
                            covariance_type='full', verbose=0,
                            init_params='kmeans'), param_grid=param_grid, scoring=gmm_bic_score, n_jobs=-1
        )
        grid_search.fit(features)
        gmm = grid_search.best_estimator_


        k_c = k_means.cluster_centers_
        g_c = gmm.means_

        for (centers, label) in zip([k_c, g_c], ["kmeans", "gmm_euc"]):
            distances = np.linalg.norm(features_test[:, None, :] - centers, axis=2)
            predicted = np.argmin(distances, axis=1)

            predicted_vecs = torch.stack([1.0 * (torch.from_numpy(predicted) == i) for i in np.arange(num_clusters)])
            gt_vecs = torch.stack([1.0 * (sub_classes_test == i) for i in np.arange(N)])
            similarity = torch.nn.functional.normalize(predicted_vecs, p=2, dim=1) @ torch.nn.functional.normalize(gt_vecs, p=2, dim=1).t()
            predicted_sorted = solve_hungarian(similarity, predicted)
            accuracy = (1.0 * (predicted_sorted == sub_classes_test)).mean() * 100
            print(f"{num_clusters}) Accuracy: {accuracy:.3f}")

            vals[label].append(accuracy)

        single_gaussians = [GaussianMixture(n_components=1, covariance_type='full') for _ in range(num_clusters)]
        for p, g_ in enumerate(single_gaussians):
            def set_param(k, param_, p_):
                if k == 0:
                    return param_[p_:p_ + 1] * 0 + 1
                elif k == 1:
                    return param_[p_:p_ + 1]
                elif k == 2:
                    return param_[p_:p_ + 1]
                elif k == 3:
                    return param_[p_:p_ + 1]
            g_._set_parameters([
                set_param(j, param, p)
                for j, param in enumerate(gmm._get_parameters())
            ])

        distances = np.stack([g_.score_samples(features_test) for g_ in single_gaussians], axis=1)
        predicted = np.argmax(distances, axis=1)

        predicted_vecs = torch.stack([1.0 * (torch.from_numpy(predicted) == i) for i in np.arange(num_clusters)])
        gt_vecs = torch.stack([1.0 * (sub_classes_test == i) for i in np.arange(N)])
        similarity = torch.nn.functional.normalize(predicted_vecs, p=2, dim=1) @ torch.nn.functional.normalize(gt_vecs,
                                                                                                               p=2,
                                                                                                               dim=1).t()
        predicted_sorted = solve_hungarian(similarity, predicted)
        accuracy = (1.0 * (predicted_sorted == sub_classes_test)).mean() * 100
        print(f"{num_clusters}) Accuracy: {accuracy:.3f}")
        vals["gmm"].append(accuracy)

    return vals





if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']

    model = get_fn_model_loader(model_name)()
    if "vgg" in model_name:
        layer_names = get_layer_names(model, [torch.nn.Conv2d])
    elif "resnet" in model_name:
        layer_names = get_layer_names(model, [torch.nn.Identity])
    elif "efficientnet" in model_name:
        layer_names = get_layer_names(model, [torch.nn.Identity])
    else:
        raise NotImplementedError

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'] if config.get("unique_wandb_ids") else None,
                   project=config['wandb_project_name'], config=config,  name=config['wandb_id'])

    num_clusters_list = [1, 2, 3, 5, 8, 10]
    x = []
    for layer_name in layer_names:
        for feature_type in [
            "eps_relevances",
                             ]:
            for group in class_groups:
                results = []
                for k in range(7):
                    print("iter", k)
                    result = main(model_name, dataset_name, layer_name, feature_type, group, num_clusters_list)
                    results.append(result)

                for method in results[0]:
                    method_results = np.array([r[method] for r in results])
                    for k, N in enumerate(num_clusters_list):
                        wandb.log({f"{method}_{layer_name}_{group}_{feature_type}_{N}": np.mean(method_results[:, k])})
                        wandb.log({f"{method}_{layer_name}_{group}_{feature_type}_{N}_err": np.std(method_results[:, k]) / np.sqrt(len(method_results[:, k]))})


