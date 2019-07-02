from torch import nn


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        shape = (batch_size,) + self.shape
        return x.view(*shape)
