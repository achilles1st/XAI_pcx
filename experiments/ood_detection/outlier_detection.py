import argparse
import os
import random

import h5py
import numpy as np
import torch as torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.covariance import ShrunkCovariance, EmpiricalCovariance
from sklearn.mixture import GaussianMixture

from utils.helper import load_config

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


def get_args():
    parser = argparse.ArgumentParser(description='Outlier Detection')
    parser.add_argument('--config_file', type=str,
                        default="configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml")
    parser.add_argument('--layer_name', type=str, default="features.28")

    return parser.parse_args()

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
def main(model_name, dataset_name, layer_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"

    experiments = [
        {"label": "Ours",
         "type": "eps_relevances"},
    ]

    for exp in experiments:
        exp['in_vecs_train'] = []
        exp['in_vecs_test'] = []
        exp['out_vecs_test'] = []
        exp['out_predicted_test'] = []

    in_sm_outputs = []
    in_logits_outputs = []

    print("Loading concept vectors...")
    classes_unique = []
    n_classes = {
        "cifar10": 10,
        "imagenet": 50,
        "birds": 200,
    }[dataset_name]

    for c in np.arange(0, n_classes):
        print("[IN] Loading class", c)
        for exp in experiments:
            data = torch.tensor(np.array(
                h5py.File(f"{path}/{exp['type']}_class_{c}_train.hdf5", "r")[layer_name]
            ))
            outputs = torch.cat(torch.load(f"{path}/outputs_class_{c}_train.pt"))
            exp['in_vecs_train'].append(data.float()[outputs.argmax(1) == c])
        classes_unique.append(c)

    for exp in experiments:
        data = torch.tensor(np.array(
            h5py.File(f"{path}/{exp['type']}_class_-1_test.hdf5", "r")[layer_name]
        ))
        exp['in_vecs_test'].append(data.float())
    in_sm_outputs.append(torch.cat(torch.load(f"{path}/outputs_class_-1_test.pt"))[:, :n_classes].softmax(1))
    in_logits_outputs.append(torch.cat(torch.load(f"{path}/outputs_class_-1_test.pt")))

    in_sm_outputs = torch.cat(in_sm_outputs, 0)
    in_logits_outputs = torch.cat(in_logits_outputs, 0)

    out_sm_outputs = []
    out_logits_outputs = []

    ood_datasets = [
        "lsun",
        "isun",
        "dtd",
        "svhn",
        # "imagenet",
        # "places365",
        # "cifar10"
    ]

    ood_predicted = []
    for ood_dataset in ood_datasets:
        path = f"results/global_features/{dataset_name}_{ood_dataset}/{model_name}"
        for c in [-1]:
            print("[OUT] Loading class", c)
            for exp in experiments:
                data = torch.tensor(np.array(
                    h5py.File(f"{path}/{exp['type']}_class_{c}_test.hdf5", "r")[layer_name]
                ))
                exp['out_vecs_test'].append(data.float())
                data = torch.cat(torch.load(f"{path}/outputs_class_{c}_test.pt"))
                exp['out_sm_test'] = torch.max(data, 1)[0].float()
            out_sm_outputs.append(data[:, :n_classes].softmax(1))
            out_logits_outputs.append(data[:, :n_classes])
            ood_predicted.append(data[:, :n_classes].argmax(1))

    print("Concept vectors loaded")

    group_lasso = EmpiricalCovariance(assume_centered=False)

    kmeans = KMeans(n_clusters=1, n_init=1)

    for exp in experiments:
        exp['means'] = torch.tensor(np.array(([kmeans.fit(v.numpy()).cluster_centers_ for v in exp['in_vecs_train']])))
        exp['stds'] = torch.stack([v.std(0) for v in exp['in_vecs_train']], 0)
        exp['precs'] = torch.stack(
            [torch.from_numpy(group_lasso.fit(v).precision_).float() for v in exp['in_vecs_train']], 0)
        exp['overall_mean'] = torch.cat(exp['in_vecs_train'], 0).mean(0)
        exp['overall_prec'] = torch.from_numpy(group_lasso.fit(torch.cat(exp['in_vecs_train'], 0)).precision_)
        exp['overall_prec'] = torch.stack([exp['overall_prec'] for _ in exp['precs']])
        exp['gms'] = [GaussianMixture(n_components=1,
                                      covariance_type='full',
                                      max_iter=100,
                                      reg_covar=1e-5,
                                      n_init=1, ).fit(v.numpy()) for v in exp['in_vecs_train']]

        exp['means'] = torch.stack([torch.from_numpy(e.means_) for e in exp['gms']], dim=0)
        exp['precs'] = torch.stack(
            [torch.from_numpy(e.precisions_) for e in exp['gms']], 0)

    def dist(vec, means, vars_):
        return torch.sqrt(((means[None] - vec[:, None, None]) ** 2 / (vars_[None, :, None] + 1e-12)).sum(-1).amin(-1))

    def dist_mahala(vec, means, precs):
        delta = (means[None] - vec[:, None, None])
        return torch.stack(
            [((torch.einsum('sbi,bij->sbj', delta[:, k], prec) * delta[:, k, ...]).sum(-1).amin(-1)) for k, prec in
             enumerate(precs)]).permute(1, 0)

    for exp in experiments:
        exp['dist_vanilla'] = []
        exp['dist_stds'] = []
        exp['dist_precs'] = []
        exp['dist_precs_maha'] = []
        exp['out_dist_vanilla'] = []
        exp['out_dist_stds'] = []
        exp['out_dist_precs'] = []
        exp['out_dist_precs_maha'] = []
        exp['dist_gm'] = []
        exp['out_dist_gm'] = []

    for c in [0]:
        print("Calculating similarities for class", c)
        for exp in experiments:
            exp['dist_vanilla'].append(dist(exp["in_vecs_test"][c], exp["means"], torch.ones_like(exp["stds"])))
            exp['dist_stds'].append(dist(exp["in_vecs_test"][c], exp["means"], exp["stds"] ** 2))
            exp['dist_precs'].append(
                dist_mahala(exp["in_vecs_test"][c], exp["means"], exp["precs"])
            )
            exp['dist_precs_maha'].append(
                dist_mahala(exp["in_vecs_test"][c], exp["means"], exp["overall_prec"][:, None])
            )

            exp['dist_gm'].append(
                torch.from_numpy(np.array([-gm.score_samples(exp["in_vecs_test"][c]) for gm in exp['gms']])).permute(1,
                                                                                                                     0)
            )

    for ood in np.arange(len(ood_datasets)):
        print("Calculating similarities for OOD", ood_datasets[ood])
        for exp in experiments:
            exp['out_dist_vanilla'].append(dist(exp["out_vecs_test"][ood], exp["means"], torch.ones_like(exp["stds"])))
            exp['out_dist_stds'].append(dist(exp["out_vecs_test"][ood], exp["means"], exp["stds"] ** 2))
            exp['out_dist_precs'].append(
                dist_mahala(exp["out_vecs_test"][ood], exp["means"], exp["precs"])
            )
            exp['out_dist_precs_maha'].append(
                dist_mahala(exp["out_vecs_test"][ood], exp["means"], exp["overall_prec"][:, None])
            )
            exp['out_dist_gm'].append(
                torch.from_numpy(np.array([-gm.score_samples(exp["out_vecs_test"][ood]) for gm in exp['gms']])).permute(
                    1, 0)
            )

    for exp in experiments:
        exp['dist_vanilla'] = torch.cat(exp['dist_vanilla'], 0)
        exp['dist_stds'] = torch.cat(exp['dist_stds'], 0)
        exp['dist_precs'] = torch.cat(exp['dist_precs'], 0)
        exp['dist_precs_maha'] = torch.cat(exp['dist_precs_maha'], 0)
        exp['dist_gm'] = torch.cat(exp['dist_gm'], 0)

    ### MAHALANOBIS

    mahalanobis = {}
    mahalanobis['in_vecs_train'] = []
    mahalanobis['in_vecs_test'] = []
    mahalanobis['out_vecs_test'] = []

    path = f"results/global_features/{dataset_name}_{dataset_name}/{model_name}"
    for c in np.arange(0, n_classes):
        print("MAHALANOBIS: [IN] Loading class", c)
        data = torch.tensor(np.array(
            h5py.File(f"{path}/mean_activations_class_{c}_train.hdf5", "r")[layer_name]
        ))
        outputs = torch.cat(torch.load(f"{path}/outputs_class_{c}_train.pt"))
        mahalanobis['in_vecs_train'].append(data.float()[outputs.argmax(1) == c])

    data = torch.tensor(np.array(
        h5py.File(f"{path}/mean_activations_class_-1_test.hdf5", "r")[layer_name]
    ))
    mahalanobis['in_vecs_test'].append(data.float())

    for ood_dataset in ood_datasets:
        path = f"results/global_features/{dataset_name}_{ood_dataset}/{model_name}"
        print("[OUT] Loading", ood_dataset)
        data = torch.tensor(np.array(
            h5py.File(f"{path}/mean_activations_class_-1_test.hdf5", "r")[layer_name]
        ))
        mahalanobis['out_vecs_test'].append(data.float())

    mahalanobis['means'] = torch.stack(([v.mean(0) for v in mahalanobis['in_vecs_train']]))
    centered = torch.cat([v - mahalanobis['means'][i] for i, v in enumerate(mahalanobis['in_vecs_train'])])
    group_lasso = EmpiricalCovariance(assume_centered=True)
    mahalanobis['prec'] = torch.from_numpy(group_lasso.fit(centered.cpu().numpy()).precision_).float()
    mahalanobis['prec'] = torch.stack([mahalanobis['prec'] for _ in range(n_classes)])

    mahalanobis['dist'] = dist_mahala(mahalanobis["in_vecs_test"][0], mahalanobis["means"][:, None],
                                      mahalanobis["prec"][:, None])

    mahalanobis['out_dist'] = []
    for ood in np.arange(len(ood_datasets)):
        print("Calculating similarities for OOD", ood_datasets[ood])
        mahalanobis['out_dist'].append(
            dist_mahala(mahalanobis["out_vecs_test"][ood], mahalanobis["means"][:, None], mahalanobis["prec"][:, None])
        )

    plt.figure(dpi=300, figsize=(4, 3))

    distance_measures = ['gm', 'precs', 'vanilla']
    ls = ['-', '--', ':']
    alpha = [1, 1, 1]
    labels = ['GMM', 'MD', 'E']

    outs_pred = in_sm_outputs[:, :n_classes].argmax(1)

    plt.figure(dpi=300, figsize=(6, 1.5 * len(ood_datasets)))

    for j, ood_dataset in enumerate(ood_datasets):

        plt.subplot(int(np.ceil(len(ood_datasets) / 2)), 2, j + 1)
        plt.title(f"{ood_dataset} OOD detection")
        AUCs = {}
        for exp in experiments:
            for i, distance_measure in enumerate(distance_measures):

                x = exp[f'dist_{distance_measure}']
                y = exp[f'out_dist_{distance_measure}'][j]
                x = torch.gather(x.t(), 0, outs_pred[None])[0]
                y = torch.gather(y.t(), 0, ood_predicted[j][None])[0]

                if distance_measure == "gm":
                    x = -x
                    y = -y
                else:
                    x = -torch.log(x)
                    y = -torch.log(y)

                FPRs, TPRs, thresholds = metrics.roc_curve(torch.cat([torch.ones_like(x),
                                                                      torch.zeros_like(y)]),
                                                           torch.cat([x, y]),
                                                           pos_label=1)

                if '' in distance_measure:
                    AUC = np.trapz(TPRs, x=FPRs) * 100
                    AUCs[f"{exp['label']}-{labels[i]}"] = AUC
                else:
                    AUC = None
                plot, = plt.plot(FPRs, TPRs, ls[i],
                                 label=f"{exp['label']}-{labels[i]} ({AUC:.2f})" if AUC is not None else None,
                                 color=plot.get_color() if i > 0 else None,
                                 alpha=alpha[i])

        # MAHALA

        x = mahalanobis[f'dist'].amin(1)
        y = mahalanobis[f'out_dist'][j].amin(1)
        x = -torch.log(x)
        y = -torch.log(y)

        FPRs, TPRs, thresholds = metrics.roc_curve(torch.cat([torch.ones_like(x),
                                                              torch.zeros_like(y)]),
                                                   torch.cat([x, y]),
                                                   pos_label=1)

        AUC = np.trapz(TPRs, x=FPRs) * 100
        AUCs[f"Mahalanobis"] = AUC

        plt.plot(FPRs, TPRs, "-", label=f"Mahalanobis ({AUC:.2f})")

        exp['dist_softmax'] = in_sm_outputs[:, :n_classes].max(1)[0] ** 1
        exp['out_dist_softmax'] = [o[:, :n_classes].max(1)[0] ** 1 for o in out_sm_outputs]

        exp['dist_energy'] = in_logits_outputs[:, :n_classes].exp().sum(-1).log()
        exp['dist_energy'] = torch.nan_to_num(exp['dist_energy'],
                                              posinf=exp['dist_energy'][exp['dist_energy'] != torch.inf].max())
        exp['out_dist_energy'] = [o[:, :n_classes].exp().sum(-1).log() for o in out_logits_outputs]
        exp['out_dist_energy'] = [
            torch.nan_to_num(o, posinf=o[o != torch.inf].max()) for o in exp['out_dist_energy']]

        for other in [
            'softmax',
            "energy"
        ]:
            x = exp[f'dist_{other}']
            y = exp[f'out_dist_{other}'][ood_datasets.index(ood_dataset)]
            FPRs, TPRs, thresholds = metrics.roc_curve(torch.cat([torch.ones_like(x),
                                                                  torch.zeros_like(y)]),
                                                       torch.cat([x, y]),
                                                       pos_label=1)

            AUC = np.trapz(TPRs, x=FPRs) * 100
            AUCs[f"{other}"] = AUC
            plot, = plt.plot(FPRs, TPRs, '-', label=other + f" ({AUC:.2f})")

        plt.legend()
        plt.xlabel("FPR (falsely classified as OOD)")
        plt.ylabel("TPR (correctly classified OOD)")
        plt.tight_layout()
        path = f"results/ood_detection/{dataset_name}/{model_name}"
        os.makedirs(path, exist_ok=True)
        torch.save(AUCs, f"{path}/{layer_name}_{ood_dataset}_aucs.pth")
    plt.show()


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    layer_name = args.layer_name

    main(model_name, dataset_name, layer_name)
