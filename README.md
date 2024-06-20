<div align="center">
<h1>Understanding the (Extra-)Ordinary: Validating Deep Model Decisions with Prototypical Concept-based Explanations</h1>
<p>
PyTorch Implementation</p>
</div>

## Description

This repository contains the code for the paper "**Understanding the (Extra-)Ordinary: Validating Deep Model Decisions with Prototypical Concept-based Explanations**''.

### Abstract 

Ensuring both transparency and safety is critical when deploying Deep Neural Networks (DNNs) in high-risk applications, such as medicine. The field of explainable AI (XAI) has proposed various methods to comprehend the decision-making processes of opaque DNNs. However, only few XAI methods are suitable of ensuring safety in practice as they heavily rely on repeated labor-intensive and possibly biased human assessment. In this work, we present a novel post-hoc concept-based XAI framework that conveys besides instance-wise (local) also class-wise (global) decision-making strategies via prototypes. What sets our approach apart is the combination of local and global strategies, enabling a clearer understanding of the (dis-)similarities in model decisions compared to the expected (prototypical) concept use, ultimately reducing the dependence on human long-term assessment. Quantifying the deviation from prototypical behavior not only allows to associate predictions with specific model sub-strategies but also to detect outlier behavior. As such, our approach constitutes an intuitive and explainable tool for model validation. We demonstrate the effectiveness of our approach in identifying out-of-distribution samples, spurious model behavior and data quality issues across three datasets (ImageNet, CUB-200, and CIFAR-10) utilizing VGG, ResNet, and EfficientNet architectures.

## Table of Contents

  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Model Training (optional)](#model-training-optional)
    - [Config Files](#config-files)
    - [Training](#training)
  - [Config Files](#config-files)
  - [Preprocessing](#preprocessing)
  - [Global Insights with Prototypes](#global-insights-with-prototypes)
  - [Evaluating Prototypes](#evaluating-prototypes)
  - [Out-of-Distribution Detection](#out-of-distribution-detection)
  - [Local Insights](#local-insights)

[//]: # (## Visuals)

## Installation

We use Python 3.8.10. To install the required packages, run:

```bash 
pip install -r requirements.txt
```

Secondly, the datasets need to be downloaded. To do so, download and extract the **CUB-200** dataset as

```bash
mkdir datasets
cd datasets
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
unzip CUB_200_2011.tgz
```

Then, we need to download the **CIFAR-10** dataset. To do so, run:

```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
cd ..
```

Further, we need to download the **ImageNet** dataset. To do so, visit
the [ImageNet website](https://image-net.org/download.php) and download the training and validation images.

Lastly, for the Out-of-Distribution Detection experiments, we need to download the following datasets:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_datasets/svhn`. Then run `python make_svhn_dataset.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_datasets/dtd`.
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/iSUN`.

Note, that we reference the OOD dataset links from the github repository of https://github.com/deeplearning-wisc/dice.

## Model Training (optional) 

**NOTE**: We provide model checkpoints! Checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1cW133sVR489TqT2OeQBm7CZ5SyCLYQdW?usp=sharing). 
Alternatively, to train models from scratch, please follow the instructions below.

Having installed the required packages and datasets,
we can begin with training the models. To do so conveniently,
we use config files to specify the model, dataset and training parameters.

### Config Files

Please adapt the config files to your setup (e.g., correct dataset paths).
To do so,
specify the config files in `config_files/*_configs.py` (replace `*` with dataset name):

Note: We suggest to use [wandb](https://wandb.ai/) to track the training progress and results.

### Training

Using the previously generated config files, we can train the models.
To do so, run:

```bash
python -m model_training.start_training --config_file "config_files/training/your_config_file.yaml"
```

## Preprocessing

In order to begin with the experiments,
we need to precompute latent activations and relevances, which, for instance, will be used for finding prototypes.
To do so, run:
```bash
python -m experiments.preprocessing.collect_features_globally.py --config_file "config_files/YOUR_CONFIG.yaml" \
                                                                 --class_id $class_id \
                                                                 --split $split \
                                                                 --dataset $dataset
```
for the ImageNet, CUB-200, and CIFAR-10 datasets, where `class_id` is the class id of the class for which we want to compute the features, `split` is the split of the dataset (train, test), and `dataset` is the dataset name (imagenet, birds, cifar10).

The whole pre-processing pipeline can be run as follows:

```bash
bash scripts/run_global_features.sh
```

For visualizing concepts, we adhere to the CRP framework and use the [CRP repository](https://github.com/rachtibat/zennit-crp).
To preprocess the data for the CRP framework, run:

```bash
python -m experiments.preprocessing.crp_run --config_file "config_files/YOUR_CONFIG.yaml"
```

## Global Insights with Prototypes

We provide several scripts to gain insights into the global model behavior.
First,
we can compute prototypes for each class, and visualize the instances that are closest to the prototypes.
To do so, run:

```bash
python -m experiments.global_understanding.plot_prototypes --config_file "config_files/YOUR_CONFIG.yaml"
```

Second,
to get additional insights into the concepts relevant for all prototypes and how they differ across prototypes, run

```bash
python -m experiments.global_understanding.plot_prototypes_with_concepts --config_file "config_files/YOUR_CONFIG.yaml"
```

Third,
to get even more detailed insights for one specific prototype with CRP localizations, run

```bash
python -m experiments.global_understanding.crp_plot_prototype --config_file "config_files/YOUR_CONFIG.yaml" \
                                                              --class_id $class_id \
                                                              --layer_name $layer_name \
                                                              --num_prototypes $num_prototypes \
                                                              --prototype $prototype
```
where `class_id` is the class id of the class for which we want to compute the features, `layer_name` is the layer name of the model, `num_prototypes` is the number of prototypes, and `prototype` is the prototype id.

Lastly,
to get an overview over class similarities with prototypes, run

```bash
python -m experiments.global_understanding.similarity_matrix_classes --config_file "config_files/YOUR_CONFIG.yaml" \
                                                                      --layer_name $layer_name
```
where `layer_name` is the layer name of the model.        


## Evaluating Prototypes 
Similarly, as for model training, we generate config files to conveniently specify the number of prototypes for evaluation. 
The config-files are located in `config_files/*` (replace `*` with the evaluation metric).
Evaluations can then be run via

```bash 
python -m experiments.faithfulness.faithfulness --config_file "config_files/YOUR_CONFIG.yaml"
python -m experiments.stability.stability --config_file "config_files/YOUR_CONFIG.yaml"
python -m experiments.sparseness.sparseness --config_file "config_files/YOUR_CONFIG.yaml"
python -m experiments.coverage.identify_subclusters --config_file "config_files/YOUR_CONFIG.yaml"
python -m experiments.outlier_detection.outlier_detection --config_file "config_files/YOUR_CONFIG.yaml"
```

## Out-of-Distribution Detection
In order to evaluate out-of-distribution detection performance, run

```bash
python -m experiments.ood_detection.outlier_detection --config_file "config_files/YOUR_CONFIG.yaml" \
                                                      --layer_name $layer_name
```
where `layer_name` is the layer name of the model. To evaluate all models and datasets, run

```bash
bash scripts/run_ood.sh
```


## Local Insights
To gain insights into the local model behavior, we can compute a concept-based explanation using CRP for any sample.
To do so, run

```bash
python -m experiments.local_understanding.crp_plot_topk_concepts --config_file "config_files/YOUR_CONFIG.yaml" \
                                                                 --layer_name $layer_name \
                                                                 --sample_id $sample_id
```
where `layer_name` is the layer name of the model, and `sample_id` is the sample id.

Alternatively,
we can also compare a single prediction with the prototype of the predicted class.
To do so, run

```bash
python -m experiments.local_understanding.crp_comparison_with_prototype --config_file "config_files/YOUR_CONFIG.yaml" \
                                                                        --layer_name $layer_name \
                                                                        --sample_id $sample_id \
                                                                        --class_id $class_id
```
where `layer_name` is the layer name of the model, `sample_id` is the sample id, and `class_id` is the class id of the predicted class.
If `sample_id` is not specified, an outlier sample is automatically chosen.