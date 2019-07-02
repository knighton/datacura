import numpy as np

from .base.curator import Curator


class RandomCurator(Curator):
    """
    Randomly samples batches out of the dataset.
    """

    def get_batch(self, mode, batch_size):
        if mode:
            split = self.dataset.train
        else:
            split = self.dataset.val
        indices = np.random.randint(0, len(split), batch_size, dtype=np.int64)
        return split[indices]

    def feedback(self, loss):
        pass
