# Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss

This is a PyTorch implementation of the [spectral contrastive learning paper](https://arxiv.org/abs/2106.04156).

Here's an example script for pretraining a resnet 18 model on cifar10 dataset, with hyper-parameter \mu=1 (see the paper for more details):

`python pretrain.py -c configs/spectral_resnet_mlp1000_norelu_cifar10_lr003_mu1.yaml --hide_progress`

Here's an example script for doing linear evaluation on the pretrained model with cifar10:

`python eval/eval_run.py --dataset cifar10 --dir PATH_TO_LOG_DIR --arch resnet18_cifar_variant1 --batch_size 256 --epochs 100 --schedule 60 80 --specific_ckpts 800.pth --opt sgd --lr 30.0 --nomlp`
