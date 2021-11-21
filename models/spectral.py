import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50


def D(z1, z2, mu=1.0):
    mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
    z2 = mask2 * z2 + (1-mask2) * F.normalize(z2, dim=1) * np.sqrt(mu)
    loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
    square_term = torch.matmul(z1, z2.T) ** 2
    loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                 z1.shape[0] / (z1.shape[0] - 1)
    return (loss_part1 + loss_part2) / mu, {"part1": loss_part1 / mu, "part2": loss_part2 / mu}


class projection_identity(nn.Module):
    def __init__(self):
        super().__init__()

    def set_layers(self, num_layers):
        pass

    def forward(self, x):
        return x


class Spectral(nn.Module):
    def __init__(self, backbone=resnet50(), mu=1.0):
        super().__init__()

        self.mu = mu
        self.backbone = backbone
        self.projector = projection_identity()

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )

    def forward(self, x1, x2, mu=1.0):
        f = self.encoder
        z1, z2 = f(x1), f(x2)
        L, d_dict = D(z1, z2, mu=self.mu)
        return {'loss': L, 'd_dict': d_dict}
