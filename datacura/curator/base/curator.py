class Curator(object):
    """
    Provides batches for training a model.
    """

    def __init__(self, dataset):
        """
        Initialize with the dataset to use.
        """
        self.dataset = dataset

    def __call__(self, mode, batch_size):
        """
        Provide a batch of (X, Y).
        """
        raise NotImplementedError

    def feedback(self, loss):
        """
        Receive news of how the model being trained liked our batch.
        """
        raise NotImplementedError
