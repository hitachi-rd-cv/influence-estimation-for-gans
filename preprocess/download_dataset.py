from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

from six.moves import urllib
from tensorflow.python.platform import gfile


def maybe_download(filename, work_directory, source_url):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """
    gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    urllib.request.urlretrieve(source_url, filepath)
    with gfile.GFile(filepath) as f:
        size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def main(dataset_name, out_dir):
    if 'mnist' in dataset_name:
        source_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
        train_file = 'train-images-idx3-ubyte.gz'
        train_labels_file = 'train-labels-idx1-ubyte.gz'
        test_file = 't10k-images-idx3-ubyte.gz'
        test_labels_file = 't10k-labels-idx1-ubyte.gz'

        maybe_download(train_file, out_dir, source_url + train_file)
        maybe_download(train_labels_file, out_dir, source_url + train_labels_file)
        maybe_download(test_file, out_dir, source_url + test_file)
        maybe_download(test_labels_file, out_dir, source_url + test_labels_file)
        return

    else:
        os.makedirs(out_dir, exist_ok=True)
