import torch
import torchvision
from .dataset_tinyimagenet import load_train_dataset, load_val_dataset
import torch.utils.data as data
import numpy as np
from PIL import Image
import os


def get_dataset(dataset, data_dir, transform, train=True, download=False):
    if dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('PATH_TO_DATASET', train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100('PATH_TO_DATASET', train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = load_train_dataset(dataset, transform) if train==True else load_val_dataset(dataset, transform)
    elif dataset == 'tiny-imagenet':
        dataset = load_train_dataset(dataset, transform) if train==True else load_val_dataset(dataset, transform)
    else:
        raise NotImplementedError

    return dataset

