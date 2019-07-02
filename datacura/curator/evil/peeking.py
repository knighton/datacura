import numpy as np
from torch.nn import functional as F

from ..base.curator import Curator


class PeekingEvilCurator(Curator):
    """
    Curator that tries to inflict high losses on the trainee model.

    Does this by simply doing a lot of dry runs and being picky.  This is the
    baseline evil curator.
    """

    def __init__(self, dataset, trainee):
        super().__init__(dataset)
        self.trainee = trainee

    def get_random_batch(self, mode, batch_size):
        if mode:
            split = self.dataset.train
        else:
            split = self.dataset.val
        indices = np.random.randint(0, len(split), batch_size, dtype=np.int64)
        return split[indices]

    def get_batch(self, mode, batch_size):
        if not mode:
            return self.get_random_batch(mode, batch_size)
        x, y_true = self.get_random_batch(mode, batch_size * 5)
        self.trainee.train()
        y_pred = self.trainee(x)
        losses = F.cross_entropy(y_pred, y_true, reduction='none')
        indices = losses.sort(0, True)[1][:batch_size]
        return x[indices], y_true[indices]

    def feedback(self, loss):
        pass
