from torch import nn

from ..layer import *


class ConvBlock(nn.Sequential):
    def __init__(self, c, stride=1):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(c, c, 3, stride, 1),
            nn.BatchNorm2d(c),
        )


class Baseline(nn.Sequential):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        assert in_shape == (3, 32, 32)
        c = 256
        super().__init__(
            nn.Conv2d(3, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            ConvBlock(c),
            ConvBlock(c),
            ConvBlock(c, 2),
            ConvBlock(c, 2),
            ConvBlock(c, 2),
            ConvBlock(c, 2),
            Flatten(),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(4 * c, c),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(c, out_dim),
        )
