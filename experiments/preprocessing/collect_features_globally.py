import argparse
import os

import h5py
import torch as torch
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from torch.utils.data import DataLoader
from tqdm import tqdm
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat, GuidedBackprop

from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
from utils.helper import load_config, get_layer_names_model
from utils.lrp_composites import EpsilonComposite, GradientComposite


def get_args():
    parser = argparse.ArgumentParser(description='Compute relevances and activations')
    parser.add_argument('--config_file', type=str,
                        default="configs/imagenet_configs/local/efficientnet_b0_Vanilla_adam_lr1_last_conv.yaml"
                        )
    parser.add_argument('--dataset', type=str, default="imagenet")
    parser.add_argument('--class_id', type=int, default=407)
    parser.add_argument('--split', type=str, default="train")
    return parser.parse_args()


def main(model_name,
         ckpt_path,
         dataset_name_in,
         dataset_name_out,
         data_path,
         split,
         class_id,
         batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "/mnt" in data_path:
        data_path_out = f"/mnt/{dataset_name_out}"
        if "birds" in dataset_name_out:
            data_path_out = f"/mnt/CUB_200_2011"
    else:
        with open("configs/local_config.yaml", "r") as stream:
            local_config = yaml.safe_load(stream)
            data_path_out = local_config[f'{dataset_name_out}_dir']

    dataset_in = get_dataset(dataset_name_in)(data_path=data_path, preprocessing=True, split="test")
    dataset = get_dataset(dataset_name_out)(data_path=data_path_out, preprocessing=True, split=split)
    # set correct transform of images
    setattr(dataset, "transform", dataset_in.transform)

    if class_id != -1:
        samples_of_class = [i for i in range(len(dataset)) if dataset.get_target(i) == class_id]
    else:
        if "imagenet" in dataset_name_out:

            filtering = lambda x: dataset.get_target(x) < 1001
            if "imagenet" in dataset_name_in:
                print("FILTERING OUT CLASSES BELOW 50 (for OOD detection experiment).")
                filtering = lambda x: dataset.get_target(x) < 50
            elif "birds" in dataset_name_in:
                print("FILTERING OUT BIRD CLASSES (for OOD detection experiment).")
                filtering = lambda x: dataset.get_target(x) not in dataset.bird_class_ids

            samples_of_class = [i for i in range(len(dataset)) if filtering(i)]
            if ("test" in split) and ("imagenet" not in dataset_name_in):
                # for OOD detection, we only need a subset of the test set
                samples_of_class = samples_of_class[::20]
        elif "places365" in dataset_name_out:
            samples_of_class = [i for i in range(len(dataset)) if dataset.get_target(i) < 1001][::10]
        else:
            samples_of_class = [i for i in range(len(dataset))]
    dataset_subset = torch.utils.data.Subset(dataset, samples_of_class)
    print("Dataset loaded")

    model = get_fn_model_loader(model_name)(ckpt_path=ckpt_path, n_class=dataset_in.num_classes)
    model = model.to(device)
    model.eval()
    print("Model loaded")

    dataloader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, num_workers=8)

    attributor = CondAttribution(model)
    canonizer = get_canonizer(model_name)
    zplus_composite = EpsilonPlusFlat(canonizer)
    eps_composite = EpsilonComposite(canonizer)
    ig_composite = GradientComposite([SequentialMergeBatchNorm()])
    guided_bp_composite = GuidedBackprop([SequentialMergeBatchNorm()])
    cc = ChannelConcept()

    layer_names = get_layer_names_model(model, model_name)

    max_activations = {}
    mean_activations = {}
    zplus_relevances = {}
    eps_relevances = {}
    ig_relevances = {}
    gbp_relevances = {}
    outputs = []
    sample_ids = []

    for i, (x, y) in enumerate(tqdm(dataloader)):
        x = x.to(device).requires_grad_()

        output = model(x)
        if split == "test":
            predicted = output.argmax(1)
            condition = [{"y": id_} for id_ in predicted]
        else:
            condition = [{"y": id_} for id_ in y]
        attr = attributor(x, condition, zplus_composite, record_layer=layer_names)

        max_activation = [attr.activations[layer][..., None].flatten(start_dim=2).amax(2) for layer in layer_names]
        mean_activation = [attr.activations[layer][..., None].flatten(start_dim=2).mean(2) for layer in layer_names]
        zplus_relevance = [cc.attribute(attr.relevances[layer], abs_norm=True) for layer in layer_names]

        attr = attributor(x, condition, eps_composite, record_layer=layer_names)
        eps_relevance = [cc.attribute(attr.relevances[layer], abs_norm=True) for layer in layer_names]

        attr = attributor(x, condition, ig_composite, record_layer=layer_names)
        ig_relevance = [cc.attribute(attr.activations[layer] * attr.relevances[layer], abs_norm=True) for layer in
                        layer_names]

        attr = attributor(x, condition, guided_bp_composite, record_layer=layer_names)
        gbp_relevance = [cc.attribute(attr.relevances[layer], abs_norm=True) for layer in layer_names]

        outputs.extend([output.detach().cpu()])
        sample_ids.extend(samples_of_class[i * batch_size:(i + 1) * batch_size])
        for k, layer in enumerate(layer_names):
            if layer not in max_activations or layer not in zplus_relevances:
                max_activations[layer] = []
                mean_activations[layer] = []
                zplus_relevances[layer] = []
                eps_relevances[layer] = []
                ig_relevances[layer] = []
                gbp_relevances[layer] = []

            max_activations[layer].append(max_activation[k].detach().cpu())
            mean_activations[layer].append(mean_activation[k].detach().cpu())
            zplus_relevances[layer].append(zplus_relevance[k].detach().cpu())
            eps_relevances[layer].append(eps_relevance[k].detach().cpu())
            ig_relevances[layer].append(ig_relevance[k].detach().cpu())
            gbp_relevances[layer].append(gbp_relevance[k].detach().cpu())
    path = f"results/global_features/{dataset_name_in}_{dataset_name_out}/{model_name}"
    os.makedirs(path, exist_ok=True)

    torch.save(outputs, f"{path}/outputs_class_{class_id}_{split}.pt")
    torch.save(sample_ids, f"{path}/sample_ids_class_{class_id}_{split}.pt")

    f = h5py.File(f"{path}/max_activations_class_{class_id}_{split}.hdf5", "w")
    for layer in layer_names:
        print(layer)
        f.create_dataset(f"{layer}", data=torch.cat(max_activations[layer], dim=0))
    f.close()

    f = h5py.File(f"{path}/mean_activations_class_{class_id}_{split}.hdf5", "w")
    for layer in layer_names:
        f.create_dataset(f"{layer}", data=torch.cat(mean_activations[layer], dim=0))
    f.close()

    f = h5py.File(f"{path}/zplus_relevances_class_{class_id}_{split}.hdf5", "w")
    for layer in layer_names:
        f.create_dataset(f"{layer}", data=torch.cat(zplus_relevances[layer], dim=0))
    f.close()

    f = h5py.File(f"{path}/eps_relevances_class_{class_id}_{split}.hdf5", "w")
    for layer in layer_names:
        f.create_dataset(f"{layer}", data=torch.cat(eps_relevances[layer], dim=0))
    f.close()

    f = h5py.File(f"{path}/ig_relevances_class_{class_id}_{split}.hdf5", "w")
    for layer in layer_names:
        f.create_dataset(f"{layer}", data=torch.cat(ig_relevances[layer], dim=0))
    f.close()

    f = h5py.File(f"{path}/gbp_relevances_class_{class_id}_{split}.hdf5", "w")
    for layer in layer_names:
        f.create_dataset(f"{layer}", data=torch.cat(gbp_relevances[layer], dim=0))
    f.close()


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name_in = config['dataset_name']
    dataset_name_out = args.dataset
    class_id = args.class_id
    batch_size = 512 if "cifar10" in dataset_name_in else 32
    data_path = config.get('data_path', None)
    ckpt_path = config.get('ckpt_path', None)
    split = args.split

    main(model_name, ckpt_path, dataset_name_in, dataset_name_out, data_path, split, class_id, batch_size)
