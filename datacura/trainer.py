import os
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .util.statistics import summarize_sorted_numpy


class EpochModeResults(object):
    def __init__(self, mode):
        self.mode = mode

        self.losses = []
        self.accuracies = []

        self.has_summary = False

        self.loss = None
        self.accuracy = None

    def update(self, loss, accuracy):
        self.has_summary = False
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def done(self):
        go = lambda x: summarize_sorted_numpy(sorted(x))
        self.loss = go(self.losses)
        self.accuracy = go(self.accuracies)
        self.has_summary = True

    def summary(self):
        if not self.has_summary:
            self.done()
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
        }


class EpochResults(object):
    def __init__(self):
        self.train = EpochModeResults(True)
        self.val = EpochModeResults(False)

    def summary(self):
        return {
            'train': self.train.summary(),
            'val': self.val.summary(),
        }


def accuracy(y_pred, y_true):
    y_pred_classes = y_pred.max(1)[1]
    return (y_pred_classes == y_true).type(torch.float32).mean().item()


def train_on_batch(model, optimizer, x, y_true):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    acc = accuracy(y_pred, y_true)
    return loss.item(), acc


def validate_on_batch(model, x, y_true):
    y_pred = model(x)
    loss = F.cross_entropy(y_pred, y_true)
    acc = accuracy(y_pred, y_true)
    return loss.item(), acc


def fit_on_epoch(device, dataset, curator, model, optimizer, train_per_epoch,
                 val_per_epoch, batch_size):
    results = EpochResults()

    each_batch = curator.get_epoch(train_per_epoch, val_per_epoch, batch_size)
    total = train_per_epoch + val_per_epoch
    each_batch = tqdm(each_batch, total=total, leave=False)

    for batch_id, (mode, x, y) in enumerate(each_batch):
        if mode:
            model.train()
            loss, acc = train_on_batch(model, optimizer, x, y)
            results.train.update(loss, acc)
        else:
            model.eval()
            loss, acc = validate_on_batch(model, x, y)
            results.val.update(loss, acc)
        curator.feedback(loss)

    return results.summary()


def epoch_results_to_line(epoch_id, x):
    t_acc = x['train']['accuracy']['mean'] * 100
    v_acc = x['val']['accuracy']['mean'] * 100
    return '%6d  %4.1f  %4.1f  %4.1f' % (epoch_id, t_acc, v_acc, t_acc - v_acc)


def fit(device, dataset, curator, model, optimizer, num_epochs, train_per_epoch,
        val_per_epoch, batch_size):
    for epoch_id in range(num_epochs):
        info = fit_on_epoch(device, dataset, curator, model, optimizer,
                            train_per_epoch, val_per_epoch, batch_size)
        line = epoch_results_to_line(epoch_id, info)
        print(line)
