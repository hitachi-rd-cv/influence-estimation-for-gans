from __future__ import division

import itertools
import json
import os
import pickle
from collections.abc import Iterable

import numpy as np
import seaborn as sns
from scipy.special import comb

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'  # for git error
sns.set()
sns.set_style('ticks')


def normalize_lists(list_):
    '''
    get 1-level nested list and returns un-nested list
    Args:
        list_:

    Returns: list

    '''
    return list(itertools.chain(*list_))


def normalize_list_recursively(list_):
    '''
    get n-level nested list and returns un-nested list
    Args:
        list_:

    Returns:

    '''
    items = []
    for item in list_:
        if isinstance(item, Iterable):
            items.extend(normalize_list_recursively(item))
        else:
            items.append(item)
    return items


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_lowest_or_highest_score_indices(scores, nsamples_low=10, nsamples_high=10):
    if len(scores) <= nsamples_low + nsamples_high:
        print('Number of scores equals to or is smaller than the total number of samples for jaccard score')
    original_indices = np.arange(len(scores))
    sorted_indices = np.argsort(scores)
    target_indices = np.concatenate([sorted_indices[:nsamples_low], sorted_indices[-nsamples_high:]])
    return np.isin(original_indices, target_indices)


def digits_to_one_hot(x, nclasses):
    x = np.array(x, dtype=np.int)
    nsamples = len(x)
    onehot = np.zeros((nsamples, nclasses), dtype=np.float)
    for i in range(nsamples):
        onehot[i, x[i]] = 1.0
    return onehot


def parse_json(path, *args, **kwargs):
    with open(path, "r") as f:
        return json.load(f, *args, **kwargs)


def dump_json(_dict, path, *args, **kwargs):
    with open(path, "w") as f:
        return json.dump(_dict, f)


def get_smallest_largest_val_indices(values, nsamples):
    sorted_indices = np.argsort(values)
    large_indices = sorted_indices[-nsamples:]
    small_indices = sorted_indices[:nsamples]
    return small_indices, large_indices


def get_minibatch_indices(nsamples, bsize, append_remainder=True, original_order=False, indices=None,
                          number_of_same_batches=1):
    nsteps, nremainders = divmod(nsamples, bsize)
    if indices is None:
        if original_order:
            perm_indices = np.arange(nsamples)
        else:
            perm_indices = np.random.permutation(np.arange(nsamples))
    else:
        perm_indices = indices
    indices_without_remainder = [perm_indices[i * bsize:(i + 1) * bsize] for i in range(nsteps)]
    if append_remainder and nremainders > 0:
        indices_remainder = [perm_indices[-nremainders:]]
        minibatch_indices = indices_without_remainder + indices_remainder
    else:
        minibatch_indices = indices_without_remainder

    if number_of_same_batches > 1:
        alt_minibatch_indices = []
        for indices in minibatch_indices:
            for _ in range(number_of_same_batches):
                alt_minibatch_indices.append(indices)
        return alt_minibatch_indices
    else:
        return minibatch_indices


def order(n):
    return str(n) + ("th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


def merge_dicts(dicts):
    return dict(itertools.chain(*[dic.items() for dic in dicts]))


def expected_indepencent_jaccard(nsamples, j_size):
    # thank you joriki! https://math.stackexchange.com/a/1770628
    n = nsamples
    m = j_size
    expected_jac = 0
    for k in np.arange(m + 1):
        expected_jac += k / (2 * m - k) * comb(m, k) * comb(n - m, m - k) / comb(n, m)
    return expected_jac


def write_str(s, fp, verbose=True):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, 'w') as f:
        f.write(s)
    if verbose:
        print(s)
