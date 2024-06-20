# Description: Script to run global features
experiment=experiments.preprocessing.collect_features_globally

dataset="cifar10"
ood_datasets=(isun lsun dtd svhn)
config_files=("configs/cifar10_configs/local/resnet18_cifar10_pretrained_True_Vanilla_sgd_lr0.01_last_conv.yaml" \
"configs/cifar10_configs/local/vgg16_cifar10_pretrained_True_Vanilla_sgd_lr0.01_features.28.yaml" \
"configs/cifar10_configs/local/efficientnet_b0_cifar10_pretrained_True_Vanilla_adam_lr0.005_last_conv.yaml"
)
for config_file in "${config_files[@]}"; do
  for class_id in {0..9}; do
    python3 -m $experiment --config_file $config_file --class_id $class_id --split train --dataset $dataset
    python3 -m $experiment --config_file $config_file --class_id $class_id --split test --dataset $dataset
  done

  python3 -m $experiment --config_file $config_file --class_id -1 --split test --dataset $dataset
  for ds in "${ood_datasets[@]}"; do
    python3 -m $experiment --config_file $config_file --class_id -1 --split test --dataset $ds
  done
done

dataset="imagenet"
ood_datasets=(isun lsun dtd svhn places365)
config_files=("configs/imagenet_configs/local/efficientnet_b0_Vanilla_adam_lr1_last_conv.yaml" \
"configs/imagenet_configs/local/resnet18_Vanilla_sgd_lr1_last_conv.yaml" \
"configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml"
)
for config_file in "${config_files[@]}"; do
  for class_id in {0..25}; do
    python3 -m $experiment --config_file $config_file --class_id $class_id --split train --dataset $dataset
  done
  for class_id in {179..204}; do
    python3 -m $experiment --config_file $config_file --class_id $class_id --split train --dataset $dataset
  done
  for class_id in {222..252}; do
    python3 -m $experiment --config_file $config_file --class_id $class_id --split train --dataset $dataset
  done
  for class_id in {389..397}; do
    python3 -m $experiment --config_file $config_file --class_id $class_id --split train --dataset $dataset
  done
  for class_id in {365..384}; do
    python3 -m $experiment --config_file $config_file --class_id $class_id --split train --dataset $dataset
  done
  for class_id in {281..293}; do
    python3 -m $experiment --config_file $config_file --class_id $class_id --split train --dataset $dataset
  done
  for class_id in {52..68}; do
    python3 -m $experiment --config_file $config_file --class_id $class_id --split train --dataset $dataset
  done
  python3 -m $experiment --config_file $config_file --class_id -1 --split test --dataset $dataset
  for ds in "${ood_datasets[@]}"; do
    python3 -m $experiment --config_file $config_file --class_id -1 --split test --dataset $ds
  done
done

dataset="birds"
ood_datasets=(isun lsun dtd svhn places365 imagenet cifar10)
config_files=("configs/birds_configs/local/resnet18_birds_pretrained_True_Vanilla_sgd_lr0.001_last_conv.yaml" \
"configs/birds_configs/local/vgg16_birds_pretrained_True_Vanilla_sgd_lr0.001_features.28.yaml" \
"configs/birds_configs/local/efficientnet_b0_birds_pretrained_True_Vanilla_adam_lr0.0005_last_conv.yaml"
)
for config_file in "${config_files[@]}"; do
  for class_id in {0..199}; do
    python3 -m $experiment --config_file $config_file --class_id $class_id --split train --dataset $dataset
    python3 -m $experiment --config_file $config_file --class_id $class_id --split test --dataset $dataset
  done

  python3 -m $experiment --config_file $config_file --class_id -1 --split test --dataset $dataset
  for ds in "${ood_datasets[@]}"; do
    python3 -m $experiment --config_file $config_file --class_id -1 --split test --dataset $ds
  done
done
