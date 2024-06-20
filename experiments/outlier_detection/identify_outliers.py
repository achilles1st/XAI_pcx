import argparse
import os
import random

import h5py
import numpy as np
import torch as torch
import umap
import wandb
from crp.helper import get_layer_names
from matplotlib import pyplot as plt
from scipy.stats import stats

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from datasets.imagenet import groups, class_groups

from models import get_fn_model_loader
from utils.helper import load_config

np.random.seed(1)
torch.manual_seed(0)
random.seed(0)

def get_args():
    parser = argparse.ArgumentParser(description='Measure Coverage')
    parser.add_argument('--config_file', type=str,
                        default="configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml"
                        )
    return parser.parse_args()



def main(model_name, dataset_name, layer_name, feature_type, group):

    path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"

    classes_to_combine = [
        *groups[group]
    ]
    # pick a subset of classes randomly
    N = 8
    pick = random.sample(classes_to_combine, N)
    outliers = random.sample([c for c in classes_to_combine if c not in pick], min(5, len(classes_to_combine) - N))
    classes_to_combine = pick

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

    features_outlier = [torch.from_numpy(np.array(
        h5py.File(f"{path}/{feature_type}_class_{CLASS_ID}_train.hdf5", "r")[layer_name][1::Nth]
    )) for CLASS_ID in outliers]

    features = torch.cat(features, dim=0)
    features_test = torch.cat(features_test, dim=0)
    features_outlier = torch.cat(features_outlier, dim=0)

    sub_classes_test = torch.cat(sub_classes_test, dim=0)

    do_plot = False
    if do_plot:
        fig, axs = plt.subplots(dpi=300, figsize=(5, 5))
        embedding = umap.UMAP(n_neighbors=15, random_state=1)
        X = embedding.fit_transform(features_test)
        x, y = X[:, 0], X[:, 1]

        xmin = x.min() - 2
        xmax = x.max() + 2
        ymin = y.min() - 2
        ymax = y.max() + 2
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values, 0.3)
        Z = np.reshape(kernel(positions).T, X.shape).T
        plt.contour(Z, extent=[xmin, xmax, ymin, ymax], cmap="Greys", alpha=0.3, extend='min', vmax=Z.max() * 1, zorder=0)

        plt.scatter(x, y, alpha=0.7, c=sub_classes_test, cmap="tab10", s=7, rasterized=True)
        plt.xticks([])
        plt.yticks([])
        plt.title("UMAP embedding")
        plt.colorbar()

        plt.tight_layout()
        os.makedirs(f"plot_files/{dataset_name}_{model_name}/embedding/", exist_ok=True)
        plt.savefig(f"plot_files/{dataset_name}_{model_name}/embedding/{dataset_name}_{model_name}_{feature_type}.svg", dpi=300)

        plt.show()

    vals = {
        "kmeans": [],
        "gmm_euc": [],
        "gmm": [],
    }

    for num_clusters in [N]:

        k_means = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(features)
        def gmm_bic_score(estimator, X):
            """Callable to pass to GridSearchCV that will use the BIC score."""
            # Make it negative since GridSearchCV expects a score to maximize
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
            distances = -torch.from_numpy(np.linalg.norm(features_test[:, None, :] - centers, axis=2)).amin(dim=1)
            distances_outlier = -torch.from_numpy(np.linalg.norm(features_outlier[:, None, :] - centers, axis=2)).amin(dim=1)

            FPRs, TPRs, thresholds = metrics.roc_curve(torch.cat([torch.ones_like(distances),
                                                                  torch.zeros_like(distances_outlier)]),
                                                       torch.cat([distances, distances_outlier]),
                                                       pos_label=1)
            AUC = np.trapz(TPRs, x=FPRs) * 100
            print(AUC)
            vals[label].append(AUC)

        distances = torch.from_numpy(gmm.score_samples(features_test))
        distances_outlier = torch.from_numpy(gmm.score_samples(features_outlier))

        FPRs, TPRs, thresholds = metrics.roc_curve(torch.cat([torch.ones_like(distances),
                                                                torch.zeros_like(distances_outlier)]),
                                                     torch.cat([distances, distances_outlier]),
                                                     pos_label=1)
        AUC = np.trapz(TPRs, x=FPRs) * 100
        print(AUC)
        vals["gmm"].append(AUC)

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

    # layer_names = [layer_names[-3]]
    x = []
    for layer_name in layer_names[:]:
        for feature_type in [
            "mean_activations",
            "max_activations",
            "eps_relevances",
            "zplus_relevances",
            "ig_relevances",
            "gbp_relevances"
                             ]:
            for group in class_groups:
                results = []
                for k in range(7):
                    print("iter", k)
                    result = main(model_name, dataset_name, layer_name, feature_type, group=group)
                    results.append(result)

                for method in results[0]:
                    method_results = [r[method] for r in results]
                    wandb.log({f"{method}_{layer_name}_{group}_{feature_type}": np.mean(method_results)})
                    wandb.log({f"{method}_{layer_name}_{group}_{feature_type}_err": np.std(method_results) / np.sqrt(len(method_results))})


