import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import folder
import torch.utils.data
import torch.utils.data.distributed
import torchvision

from .loader import TwoCropsTransform, GaussianBlur


def get_dataset_path():
    if os.path.exists('PATH_TO_DATASET'):
        return 'PATH_TO_DATASET'
    elif os.path.exists('PATH_TO_DATASET'):
        return 'PATH_TO_DATASET'
    elif os.path.exists('PATH_TO_DATASET'):
        return 'PATH_TO_DATASET'


data_path_dict = {
    'imagenet': get_dataset_path(),
    'tiny-imagenet': 'PATH_TO_DATASET',
    'cifar10': 'PATH_TO_DATASET',
    'cifar100': 'PATH_TO_DATASET'
}

crop_size_dict = {
    'imagenet': 224,
    'tiny-imagenet': 64,
    'cifar10': 32,
    'cifar100': 32
}

resize_size_dict = {
    'imagenet': 256,
    'tiny-imagenet': 74,
    'cifar10': 40,
    'cifar100': 40
}

num_classes_dict = {
    'imagenet': 1000,
    'tiny-imagenet': 200,
    'cifar10': 10,
    'cifar100': 100
}
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def obtain_aug(dataset, data_aug, aug_plus): 
    crop_size = crop_size_dict[dataset]   
    if data_aug == 'pretrain':
        if aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
            # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
            augmentation = [
                transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        train_transform = TwoCropsTransform(transforms.Compose(augmentation))
    elif data_aug == 'standard':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif data_aug == 'mocov1':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif data_aug == 'mocov2':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(resize_size_dict[dataset]),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])
    return train_transform


def load_train_dataset(dataset, tranform):
    traindir = os.path.join(data_path_dict[dataset], 'train')
    return folder.ImageFolder(traindir, tranform)


def load_train(dataset, num_per_class, distributed, batch_size, workers, 
        aug_plus=False, orig_aug=None, data_aug='pretrain', mode='train', random_labels=None):
    '''
    data_aug:
        if pretrain, apply contrastive learning data augmentation (returning 2 crops),
        if standard, simply choose a single random crop (for linear classification).
        if off, choose center crop (no data augmentation applied).
    '''
    
    data_path = data_path_dict[dataset]
    assert mode in ['train', 'val']

    if dataset == 'cifar10':
        train_transform = obtain_aug(dataset, data_aug, aug_plus)
        train_dataset = torchvision.datasets.CIFAR10(data_path_dict['cifar10'], train=True, transform=train_transform, download=False)
    elif dataset == 'cifar100':
        train_transform = obtain_aug(dataset, data_aug, aug_plus)
        train_dataset = torchvision.datasets.CIFAR100(data_path_dict['cifar100'], train=True, transform=train_transform, download=False)
    else:
        traindir = os.path.join(data_path, mode)

        train_transform = obtain_aug(dataset, data_aug, aug_plus)
        if orig_aug is not None:
            orig_aug = obtain_aug(dataset, orig_aug, False)
        train_dataset = SubsetImageFolder_NoAug(
            traindir, orig_transform=orig_aug,
            transform=train_transform, num_per_class=num_per_class,
            random_labels=random_labels)
        print('train dataset size is', len(train_dataset))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(not distributed),
        num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=data_aug == 'pretrain')
    
    return train_sampler, train_loader


def load_val_dataset(dataset, tranform):
    valdir = os.path.join(data_path_dict[dataset], 'val')
    return folder.ImageFolder(valdir, tranform)


def load_val_loader(dataset, batch_size, workers):
    val_transform = transforms.Compose([
            transforms.Resize(resize_size_dict[dataset]),
            transforms.CenterCrop(crop_size_dict[dataset]),
            transforms.ToTensor(),
            normalize
        ])
    if dataset == 'cifar10':
        val_dataset = torchvision.datasets.CIFAR10(data_path_dict['cifar10'], train=False, transform=val_transform, download=False)
    elif dataset == 'cifar100':
        val_dataset = torchvision.datasets.CIFAR100(data_path_dict['cifar100'], train=False, transform=val_transform, download=False)
    else:
        valdir = os.path.join(data_path_dict[dataset], 'val')
        val_dataset = folder.ImageFolder(valdir, val_transform)
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


def get_loaders(dataset, num_per_class, distributed, batch_size, workers, 
        aug_plus=False, data_aug='standard', train_mode='train', random_labels=None):
    _, train_loader = load_train(dataset, num_per_class, distributed, batch_size, workers, 
        aug_plus=aug_plus, data_aug=data_aug, mode=train_mode, random_labels=random_labels)
    val_loader = load_val_loader(dataset, batch_size, workers)
    return train_loader, val_loader


class SubsetImageFolder_NoAug(folder.DatasetFolder):
    """
    Data loader that loads only a subset of the samples
    """
    def __init__(self, root, orig_transform=None, transform=None, target_transform=None, num_per_class=None,
                 loader=folder.default_loader, extensions=folder.IMG_EXTENSIONS,
                 random_labels=None):
        super(folder.DatasetFolder, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, num_per_class)
        if random_labels is not None:
            samples = [(inst[0], rl) for (inst, rl) in zip(samples, random_labels)]
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.orig_transform = orig_transform

        self.use_random_labels = random_labels is not None
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        if self.orig_transform is None:
            return super(SubsetImageFolder_NoAug, self).__getitem__(index)
        else:
            path, target = self.samples[index]
            orig_sample = self.loader(path)
            if self.target_transform is not None:
                target = self.target_transform(target)
            orig_sample = self.orig_transform(orig_sample)
            return orig_sample, target


class SubsetImageFolder(folder.DatasetFolder):
    """
    Data loader that loads only a subset of the samples
    """
    def __init__(self, root, orig_transform=None, transform=None, target_transform=None, num_per_class=None,
                 loader=folder.default_loader, extensions=folder.IMG_EXTENSIONS,
                 random_labels=None):
        super(folder.DatasetFolder, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, num_per_class)
        if random_labels is not None:
            samples = [(inst[0], rl) for (inst, rl) in zip(samples, random_labels)]
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.orig_transform = orig_transform

        self.use_random_labels = random_labels is not None
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        if self.orig_transform is None:
            return super(SubsetImageFolder, self).__getitem__(index)
        else:
            path, target = self.samples[index]
            orig_sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(orig_sample.copy())
            else:
                sample = orig_sample.copy()
            if self.target_transform is not None:
                target = self.target_transform(target)
            orig_sample = self.orig_transform(orig_sample)
            return (sample, orig_sample), target


def make_dataset(directory, class_to_idx, extensions, num_per_class):
    instances = []
    directory = os.path.expanduser(directory)
    def is_valid_file(x):
        return folder.has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        num_added = 0
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            if num_added >= num_per_class:
                break
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    num_added += 1
                    if num_added >= num_per_class:
                        break
    return instances
