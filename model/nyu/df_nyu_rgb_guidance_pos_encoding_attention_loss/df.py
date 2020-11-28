import torch.nn as nn
import torch
import torch.nn.functional as F
from unet import Unet
from config import config

class Displacement_Field(nn.Module):
    def __init__(self):
        super(Displacement_Field, self).__init__()
        self.displacement_net = Unet(n_channels=1, rgb_channels=1, n_classes=2)
        self.theta = torch.Tensor([[1, 0, 0],
                                   [0, 1, 0]])
        self.theta = self.theta.view(-1, 2, 3)

    def forward(self, rgb, x):
        max_disp = .9
        displacement_map = self.displacement_net(rgb, x)
        output = []

        displacement_map = displacement_map / 320
        displacement_map = displacement_map.clamp(min=-max_disp, max=max_disp)

        theta = self.theta.repeat(x.size()[0], 1, 1)
        grid = F.affine_grid(theta, x.size()).cuda()
        grid = (grid + displacement_map.transpose(1,2).transpose(2,3)).clamp(min=-1, max=1)
        x = F.grid_sample(x, grid)
        return x

