from argparse import ArgumentParser
import torch
from torch import optim

from .curator import RandomCurator
from .dataset import *
from . import model as models
from .trainer import fit


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--device', type=str, default='cuda:0')
    a.add_argument('--dataset', type=str, default='cifar100')
    a.add_argument('--curator', type=str, default='random')
    a.add_argument('--model', type=str, default='Baseline')
    a.add_argument('--optimizer', type=str, default='Adam')
    a.add_argument('--num_epochs', type=int, default=1000)
    a.add_argument('--train_per_epoch', type=int, default=100)
    a.add_argument('--val_per_epoch', type=int, default=50)
    a.add_argument('--batch_size', type=int, default=32)
    return a.parse_args()


def parse_flag_kwarg_value(s):
    try:
        return int(s)
    except:
        pass
    try:
        return float(s)
    except:
        pass
    return s


def parse_flag_kwargs(s):
    kwargs = {}
    ss = s.split(',')
    for s in ss:
        index = s.index('=')
        k = s[:index]
        v = parse_flag_kwarg_value(s[index + 1:])
        kwargs[k] = v
    return kwargs


def parse_flag(s):
    index = s.find(':')
    if index == -1:
        class_name = s
        kwargs = {}
    else:
        class_name = s[:index]
        kwargs = parse_flag_kwargs(s[index + 1:])
    return class_name, kwargs


def get_dataset(name, device):
    if name == 'cifar10':
        return load_cifar10(device)
    elif name == 'cifar20':
        return load_cifar20(device)
    elif name == 'cifar100':
        return load_cifar100(device)
    else:
        assert False


def get_curator(name, dataset):
    if name == 'random':
        return RandomCurator(dataset)
    else:
        assert False


def get_model(name, in_shape, out_classes):
    klass = getattr(models, name)
    return klass(in_shape, out_classes)


def get_optimizer(name, model):
    class_name, kwargs = parse_flag(name)
    klass = getattr(optim, class_name)
    return klass(model.parameters(), **kwargs)


def main(f):
    device = torch.device(f.device)
    dataset = get_dataset(f.dataset, device)
    curator = get_curator(f.curator, dataset)
    model = get_model(f.model, dataset.get_x_shape(), dataset.num_classes())
    model.to(device)
    optimizer = get_optimizer(f.optimizer, model)
    fit(device, dataset, curator, model, optimizer, f.num_epochs,
        f.train_per_epoch, f.val_per_epoch, f.batch_size)


if __name__ == '__main__':
    main(parse_flags())
