import numpy as np
import torch

from .clf_dataset import ClfDataset, ClfDatasetSplit


class RamImgClfDatasetSplit(ClfDatasetSplit):
    """
    In-memory image classification dataset split.
    """

    def __init__(self, x, y):
        super().__init__()

        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 4
        assert x.shape[1] in {1, 3}
        assert x.dtype == np.uint8

        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1
        assert issubclass(x.dtype.type, np.integer)

        assert x.shape[0] == y.shape[0]

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index, device):
        x = self.x[index]
        x = torch.from_numpy(x, dtype=torch.float32, device=device)
        x /= 255
        y = self.y[index]
        y = torch.from_numpy(y, dtype=torch.int64, device=device)
        return x, y

    def x_shape(self):
        return self.x.shape[1:]

    def y_shape(self):
        return self.y.shape[1:]


class RamImgClfDataset(ClfDataset):
    """
    Image classificaiton dataset kept entirely in CPU memory as numpy arrays.
    """

    def __init__(self, train, val, class_names):
        train = RamImgClfDatasetSplit(*train)
        val = RamImgClfDatasetSplit(*val)
        super().__init__(train, val, class_names)
