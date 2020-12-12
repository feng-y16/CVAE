import os
import gzip
import tarfile
import zipfile
import math
import numpy as np
import six
import urllib
import pickle
from tensorflow.examples.tutorials.mnist import input_data
import pdb


def download_dataset(url, path):
    if not os.path.exists(path):
        print('Downloading data from %s' % url)
        urllib.request.urlretrieve(url, path)
    else:
        print('File already exists.')


def load_mnist_datasets(path, one_hot=True):
    """
    Loads the MNIST dataset.

    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.

    :return: The MNIST dataset.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    download_dataset('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                     path + '/train-images-idx3-ubyte.gz')
    download_dataset('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                     path + '/train-labels-idx1-ubyte.gz')
    download_dataset('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                     path + '/t10k-images-idx3-ubyte.gz')
    download_dataset('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
                     path + '/t10k-labels-idx1-ubyte.gz')
    datasets = input_data.read_data_sets(path, one_hot=one_hot)
    return datasets
