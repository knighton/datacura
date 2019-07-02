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

    def get_x_dtype(self):
        """
        Get X dtype.
        """
        raise NotImplementedError

    def get_y_shape(self):
        """
        Get Y sample shape.
        """
        raise NotImplementedError

    def get_y_dtype(self):
        """
        Get Y dtype.
        """
        raise NotImplementedError

    def get_shapes(self):
        """
        Get X and Y sample shapes.
        """
        return self.get_x_shape(), self.get_y_shape()

    def get_dtypes(self):
        """
        Get X and Y dtypes.
        """
        return self.get_x_dtype(), self.get_y_dtype()


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

    def get_x_dtype(self):
        """
        Get X dtype.
        """
        return self.train.get_x_dtype()

    def get_y_shape(self):
        """
        Get Y sample shape.
        """
        return self.train.y_shape()

    def get_y_dtype(self):
        """
        Get Y dtype.
        """
        return self.train.get_y_dtype()

    def get_shapes(self):
        """
        Get X and Y sample shapes.
        """
        return self.train.get_shapes()

    def get_dtypes(self):
        """
        Get X and Y dtypes.
        """
        return self.train.get_dtypes()
