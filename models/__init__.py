from .spectral import Spectral
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2
from .backbones import resnet18_cifar_variant1_mlp1000_norelu
from .backbones import resnet50_mlp8192_norelu_3layer


def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):
    if model_cfg.name == 'spectral':
        if "mu" not in model_cfg.__dict__:
            model_cfg.mu = 1.0
        model = Spectral(get_backbone(model_cfg.backbone), mu=model_cfg.mu)

    else:
        raise NotImplementedError
    return model






