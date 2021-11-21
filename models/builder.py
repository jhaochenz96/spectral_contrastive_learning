import torch
import torch.nn as nn
import torchvision.models as models
import sys
sys.path.append('../')
import utils
from models.backbones import resnet18_cifar_variant1
from models.backbones.cifar_resnet_1_mlp_norelu import resnet18_cifar_variant1_mlp1000_norelu
from models.backbones.resnet_mlp_norelu_3layer import resnet50_mlp8192_norelu_3layer


def get_model(out_dim, arch='resnet50'):
    try:
        base_model = models.__dict__[arch]
    except:
        base_model = globals()[arch]

    model = base_model(num_classes=out_dim)
    return model


def load_checkpoint(model, state_dict, fname, load_pretrained_head=False, args=None, nomlp=False):
    print("=> loading checkpoint '{}'".format(fname))
    state_dict = utils.fix_dataparallel_keys(state_dict)

    # Rename pre-trained keys
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer. However,
        # if load pretrained head is set, then also retain the fc weights.
        if nomlp:
            if k.startswith('backbone.') and (load_pretrained_head or (
                (not load_pretrained_head and not k.startswith('backbone.fc')))) and \
                    (load_pretrained_head or (
                            (not load_pretrained_head and not k.startswith('backbone.proj_resnet_layer1')))) and \
                    (load_pretrained_head or (
                            (not load_pretrained_head and not k.startswith('backbone.proj_resnet_layer2')))):
                # remove prefix
                state_dict[k[len('backbone.'):]] = state_dict[k]
        else:
            if k.startswith('backbone.') and \
                    (load_pretrained_head or (not load_pretrained_head and not k.startswith('backbone.fc'))):
                # remove prefix
                state_dict[k[len('backbone.'):]] = state_dict[k]
        # delete renamed or unused keys
        del state_dict[k]
    
    if args is not None:
        args.start_epoch = 0
    if load_pretrained_head:
        model.load_state_dict(state_dict, strict=True) # shouldn't have missing keys
    else:
        msg = model.load_state_dict(state_dict, strict=False)
        if not isinstance(model.fc, nn.Identity):
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print("=> loaded pre-trained model '{}'".format(fname))

