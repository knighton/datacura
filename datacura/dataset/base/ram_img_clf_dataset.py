import numpy as np
import torch

from .clf_dataset import ClfDataset, ClfDatasetSplit


class RamImgClfDatasetSplit(ClfDatasetSplit):
    """
    In-memory image classification dataset split.
    """

    def __init__(self, x, y, device):
        super().__init__(device)

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

    def __getitem__(self, index):
        x = self.x[index]
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x /= 255
        y = self.y[index]
        y = torch.tensor(y, dtype=torch.int64, device=self.device)
        return x, y

    def get_x_shape(self):
        return self.x.shape[1:]

    def get_x_dtype(self):
        return torch.float32

    def get_y_shape(self):
        return self.y.shape[1:]


class RamImgClfDataset(ClfDataset):
    """
    Image classificaiton dataset kept entirely in CPU memory as numpy arrays.
    """

    def __init__(self, train, val, class_names, device):
        x, y = train
        train = RamImgClfDatasetSplit(x, y, device)
        x, y = val
        val = RamImgClfDatasetSplit(x, y, device)
        super().__init__(train, val, class_names)
