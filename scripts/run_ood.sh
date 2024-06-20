# VGG MODELS
config_file=configs/birds_configs/local/vgg16_birds_pretrained_True_Vanilla_sgd_lr0.001_features.28.yaml
python3 -m experiments.ood_detection.outlier_detection --config_file $config_file --layer_name features.28
config_file=configs/cifar10_configs/local/vgg16_cifar10_pretrained_True_Vanilla_sgd_lr0.01_features.28.yaml
python3 -m experiments.ood_detection.outlier_detection --config_file $config_file --layer_name features.28
config_file=configs/imagenet_configs/local/vgg16_Vanilla_sgd_lr1_features.28.yaml
python3 -m experiments.ood_detection.outlier_detection --config_file $config_file --layer_name features.28

# RESNET MODELS
config_file=configs/birds_configs/local/resnet18_birds_pretrained_True_Vanilla_sgd_lr0.001_last_conv.yaml
python3 -m experiments.ood_detection.outlier_detection --config_file $config_file --layer_name last_conv
config_file=configs/cifar10_configs/local/resnet18_cifar10_pretrained_True_Vanilla_sgd_lr0.01_last_conv.yaml
python3 -m experiments.ood_detection.outlier_detection --config_file $config_file --layer_name last_conv
config_file=configs/imagenet_configs/local/resnet18_Vanilla_sgd_lr1_last_conv.yaml
python3 -m experiments.ood_detection.outlier_detection --config_file $config_file --layer_name last_conv

# EFFICIENTNET MODELS
config_file=configs/birds_configs/local/efficientnet_b0_birds_pretrained_True_Vanilla_adam_lr0.0005_last_conv.yaml
python3 -m experiments.ood_detection.outlier_detection --config_file $config_file --layer_name last_conv
config_file=configs/cifar10_configs/local/efficientnet_b0_cifar10_pretrained_True_Vanilla_adam_lr0.005_last_conv.yaml
python3 -m experiments.ood_detection.outlier_detection --config_file $config_file --layer_name last_conv
config_file=configs/imagenet_configs/local/efficientnet_b0_Vanilla_adam_lr1_last_conv.yaml
python3 -m experiments.ood_detection.outlier_detection --config_file $config_file --layer_name last_conv