from .dataset import Dataset, DatasetSplit


class ClfDatasetSplit(DatasetSplit):
    """
    Classification dataset split base class.
    """
    pass


class ClfDataset(Dataset):
    """
    Classification dataset base class.
    """

    def __init__(self, train, val, class_names):
        super().__init__(train, val)

        assert isinstance(class_names, list)
        assert len(class_names) == len(set(class_names))
        for s in class_names:
            assert s
            assert isinstance(s, str)

        self.class_names = class_names

    def get_class_names(self):
        """
        Get string class names (in order).
        """
        return self.class_names

    def num_classes(self):
        """
        Get the number of classes.
        """
        return len(self.class_names)

