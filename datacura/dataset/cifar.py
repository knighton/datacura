import numpy as np
import os
import pickle
import tarfile
from tqdm import tqdm

from .base import DATASET_DIR, download_if_missing


CIFAR10_REMOTE = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_NAME = 'cifar10'

CIFAR100_REMOTE = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR100_NAME = 'cifar100'


def _load_cifar10_splits(tar, verbose):
    if verbose == 2:
        bar = tqdm(total=5, leave=False)

    train = []
    val = None
    for info in tar.getmembers():
        if not info.isreg():
            continue

        if info.path.startswith('cifar-10-batches-py/data_batch_'):
            is_train = True
        elif info.path == 'cifar-10-batches-py/test_batch':
            is_train = False
        else:
            continue

        data = tar.extractfile(info).read()
        obj = pickle.loads(data, encoding='bytes')
        x = obj[b'data']
        x = np.array(x, np.uint8)
        x = x.reshape(-1, 3, 32, 32)
        y = obj[b'labels']
        y = np.array(y, np.uint8)

        if is_train:
            train.append((info.path, x, y))
        else:
            val = x, y

        if verbose == 2:
            bar.update(1)

    if verbose == 2:
        bar.close()

    train.sort()
    _, xx, yy = zip(*train)
    x = np.concatenate(xx, 0)
    y = np.concatenate(yy, 0)
    train = x, y

    return train, val


def _load_cifar10_class_names(tar):
    path = 'cifar-10-batches-py/batches.meta'
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    labels = obj[b'label_names']
    return list(map(lambda s: s.decode('utf-8'), labels))


def load_cifar10(verbose=2):
    """
    Load the CIFAR-10 dataset.
    """
    dataset_dir = os.path.join(DATASET_DIR, CIFAR10_NAME)
    local = os.path.join(dataset_dir, os.path.basename(CIFAR10_REMOTE))
    download_if_missing(CIFAR10_REMOTE, local, verbose)
    tar = tarfile.open(local, 'r:gz')
    splits = _load_cifar10_splits(tar, verbose)
    class_names = _load_cifar10_class_names(tar)
    tar.close()
    return splits, class_names


def _load_cifar100_split(tar, classes, split):
    path = 'cifar-100-python/%s' % split
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    x = obj[b'data']
    x = np.array(x, np.uint8)
    x = x.reshape(-1, 3, 32, 32)
    if classes == 20:
        key = b'coarse_labels'
    elif classes == 100:
        key = b'fine_labels'
    else:
        assert False
    y = obj[key]
    y = np.array(y, np.uint8)
    return x, y


def _load_cifar100_class_names(tar, classes):
    info = tar.getmember('cifar-100-python/meta')
    data = tar.extractfile(info).read()
    obj = pickle.loads(data, encoding='bytes')
    if classes == 20:
        key = b'coarse_label_names'
    elif classes == 100:
        key = b'fine_label_names'
    else:
        assert False
    labels = obj[key]
    return list(map(lambda s: s.decode('utf-8'), labels))


def _load_cifar100(classes, verbose):
    dataset_dir = os.path.join(DATASET_DIR, CIFAR100_NAME)
    local = os.path.join(dataset_dir, os.path.basename(CIFAR100_REMOTE))
    download_if_missing(CIFAR100_REMOTE, local, verbose)
    tar = tarfile.open(local, 'r:gz')
    train = _load_cifar100_split(tar, classes, 'train')
    val = _load_cifar100_split(tar, classes, 'test')
    class_names = _load_cifar100_class_names(tar, classes)
    tar.close()
    return (train, val), class_names


def load_cifar20(verbose=2):
    """
    Load the CIFAR-20 dataset.
    """
    return _load_cifar100(20, verbose)


def load_cifar100(verbose=2):
    """
    Load the CIFAR-100 dataset.
    """
    return _load_cifar100(100, verbose)


def load_cifar(classes, verbose=2):
    """
    Load a CIFAR dataset, specifying the number of classes (10, 20, or 100).
    """
    if classes == 10:
        return load_cifar10(verbose)
    elif classes == 20:
        return load_cifar20(verbose)
    elif classes == 100:
        return load_cifar100(verbose)
    else:
        assert False