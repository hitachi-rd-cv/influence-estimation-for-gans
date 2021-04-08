import gzip

import numpy as np
import tensorflow as tf

from modules.dataset import dump_dataset
from modules.tf_ops import create_multivariate_gaussian_distribution
from modules.utils import digits_to_one_hot


def extract_gzip(filename, num_data, head_size, data_size):
    with gzip.open(filename) as bytestream:
        bytestream.read(head_size)
        buf = bytestream.read(data_size * num_data)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
    return data


def main(dataset_name, dataset_dir, out_dir, nsamp_train, nsamp_test, nsamp_valid,
         nclasses=10, permute_train=True, permute_test=True, seed=547, option=None):
    # set random seed
    np.random.seed(seed)

    if dataset_name == 'mnist':
        assert nclasses == 10
        data = extract_gzip(dataset_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
        trX = data.reshape((60000, 28, 28, 1))
        trX = trX / 255. * 2 - 1  # normalize to -1 to 1

        data = extract_gzip(dataset_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
        trY = data.reshape((60000))
        trY = np.asarray(trY).astype(np.int)
        tr_labels = np.asarray([str(x) for x in trY])

        data = extract_gzip(dataset_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
        teX = data.reshape((10000, 28, 28, 1))
        teX = teX / 255. * 2 - 1  # normalize to -1 to 1

        data = extract_gzip(dataset_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
        teY = data.reshape((10000))
        teY = np.asarray(teY).astype(np.int)
        te_labels = np.asarray([str(x) for x in teY])  # label strings

    elif 'mulvargaussian' in dataset_name:
        assert nclasses == 1
        mulvargauss = create_multivariate_gaussian_distribution(np.float64)
        samples = mulvargauss.sample()
        nsamp_tr_va = nsamp_train + nsamp_valid
        trY = np.zeros(nsamp_tr_va)
        teY = np.zeros(nsamp_test)
        tr_labels = np.asarray([str(x) for x in trY])
        te_labels = np.asarray([str(x) for x in teY])
        trX = np.zeros([nsamp_tr_va, 2])
        teX = np.zeros([nsamp_test, 2])
        with tf.Session() as sess:
            for i in range(nsamp_tr_va):
                trX[i] = sess.run(samples)  # generate training samples
            for i in range(nsamp_test):
                teX[i] = sess.run(samples)  # generate training samples

    else:
        raise ValueError(dataset_name)

    # to onehot
    trY_onehot, teY_onehot = digits_to_one_hot(trY, nclasses), digits_to_one_hot(teY, nclasses)

    # permutation indice
    if permute_train:
        tr_indices_all = np.random.permutation(np.arange(len(trY)))
    else:
        tr_indices_all = np.arange(len(trY))
    if permute_test:
        te_indices_all = np.random.permutation(np.arange(len(teY)))
    else:
        te_indices_all = np.arange(len(teY))

    # sampling
    assert nsamp_train + nsamp_valid <= len(tr_indices_all)
    tr_indices_selected = tr_indices_all[:nsamp_train]
    va_indices_selected = tr_indices_all[nsamp_train:nsamp_train + nsamp_valid]
    te_indices_selected = te_indices_all[:nsamp_test]

    dump_dataset(trX[tr_indices_selected], trY_onehot[tr_indices_selected], tr_labels[tr_indices_selected], 'train',
                 out_dir)
    dump_dataset(trX[va_indices_selected], trY_onehot[va_indices_selected], tr_labels[va_indices_selected], 'valid',
                 out_dir)
    dump_dataset(teX[te_indices_selected], teY_onehot[te_indices_selected], te_labels[te_indices_selected], 'test',
                 out_dir)
