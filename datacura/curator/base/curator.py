import numpy as np


class Curator(object):
    """
    Provides batches for training a model.
    """

    def __init__(self, dataset):
        """
        Initialize with the dataset to use.
        """
        self.dataset = dataset

    def get_batch(self, mode, batch_size):
        """
        Provide a batch of (X, Y).
        """
        raise NotImplementedError

    def get_epoch(self, num_train, num_val, batch_size):
        """
        Provide an epoch of batches.
        """
        modes = [1] * num_train + [0] * num_val
        np.random.shuffle(modes)
        for mode in modes:
            x, y = self.get_batch(mode, batch_size)
            yield mode, x, y

    def feedback(self, loss):
        """
        Receive news of how the model being trained liked our batch.
        """
        raise NotImplementedError
