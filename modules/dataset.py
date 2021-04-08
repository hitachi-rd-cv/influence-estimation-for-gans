from __future__ import print_function

import os

import numpy as np
import torch.utils.data as data
from prettytable import PrettyTable


class MyDataset(data.Dataset):
    def __init__(self, feed_dict, labels=None, name='no_name'):
        assert feed_dict.__class__ == dict, 'inputs must be dict type'
        self.feed_dict = feed_dict
        self.sample_size = self._get_sample_size(feed_dict)
        self.name = name
        if labels is not None:
            assert len(labels) == self.sample_size
        self.labels = labels

    @staticmethod
    def _get_sample_size(feed_dict):
        sample_size = None
        for v in feed_dict.values():
            size_tmp = v.shape[0]
            if sample_size is None:
                sample_size = size_tmp
            assert size_tmp == sample_size, 'Sample sizes are inconsistent'
        return sample_size

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        '''
        returns the feed dict for Session.run
        Args:
            index: integer or list of integers

        Returns: dictionary of feed_dict for tf.Session.run

        '''
        if isinstance(index, (int, np.integer)):
            key_ = [index]
        else:
            key_ = index

        return {k: v.__getitem__(key_) for k, v in self.feed_dict.items()}

    def __repr__(self):
        unique_labels = np.unique(self.labels)
        pt = PrettyTable(['dataset', *unique_labels, 'total'])
        nsamples_of_classes = [np.sum(self.labels == label).astype(np.int) for label in unique_labels]
        row = [self.name]
        row.extend(nsamples_of_classes)
        row.append(self.sample_size)
        pt.add_row(row)
        return pt.__str__()

    def update(self, feed_dict, *args, **kwargs):
        assert isinstance(feed_dict, dict), 'others must be dict type'
        assert all([len(x) == self.sample_size for x in feed_dict.values()]), 'inconsistent sample length'
        self.feed_dict.update(feed_dict)


def load_and_create_dataset(dataset_dir, x_ph, y_ph, name=None, print_info=True):
    X, Y = load_dataset(dataset_dir)
    labels = load_labels(dataset_dir)

    name_ = name or os.path.basename(dataset_dir)

    dataset = MyDataset({x_ph: X, y_ph: Y}, labels, name_)
    if print_info:
        print(dataset)

    return dataset


def load_dataset(dataset_dir, flatten=False):
    X = np.load(dataset_dir + "/x.npy")
    Y = np.load(dataset_dir + "/y.npy")
    if flatten:
        return X.reshape([len(X), -1]), Y
    else:
        return X, Y


def load_labels(dataset_dir, train=True):
    if train:
        labels = np.load(dataset_dir + "/labels.npy")
    else:
        labels = np.load(dataset_dir + "/labels.npy")
    return labels


def dump_dataset(X, Y_onehot, labels, name, out_dir):
    assert len(X) == len(Y_onehot) == len(labels)
    dataset_dir = os.path.join(out_dir, name)
    os.makedirs(dataset_dir, exist_ok=True)
    np.save(dataset_dir + "/x.npy", X)
    np.save(dataset_dir + "/y.npy", Y_onehot)
    np.save(dataset_dir + "/labels.npy", labels)
    print("Extracted {} dataset is placed in {}".format(name, out_dir))
