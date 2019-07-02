class DatasetSplit(object):
    """
    Dataset split base class.
    """

    def __len__(self):
        """
        Get the number of samples.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        """
        Get the sample(s) at the given index(es).
        """
        raise NotImplementedError

    def get_x_shape(self):
        """
        Get X sample shape.
        """
        raise NotImplementedError

    def get_y_shape(self):
        """
        Get Y sample shape.
        """
        raise NotImplementedError

    def get_shape(self):
        """
        Get X and Y sample shapes.
        """
        return self.x_shape(), self.y_shape()


class Dataset(object):
    """
    Dataset base class.
    """

    def __init__(self, train, val):
        """
        Initialize with training and validation splits.
        """
        self.train = train
        self.val = val

    def get_x_shape(self):
        """
        Get X sample shape.
        """
        return self.train.x_shape()

    def get_y_shape(self):
        """
        Get Y sample shape.
        """
        return self.train.y_shape()

    def get_shape(self):
        """
        Get X and Y sample shapes.
        """
        return self.train.get_shape()
