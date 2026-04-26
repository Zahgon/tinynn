"""Common datasets"""

import gzip
import os
import pickle
import struct
import tarfile

import numpy as np

from tinynn.utils.downloader import download_url


class Dataset:

    def __init__(self, data_dir, **kwargs):
        self._train_set = None
        self._valid_set = None
        self._test_set = None

        self._save_paths = [os.path.join(data_dir, url.split("/")[-1])
                            for url in self._urls]

        self._download()
        self._parse(**kwargs)  # lgtm [py/init-calls-subclass]

    def _download(self):
        pass

    def _parse(self, **kwargs):
        raise NotImplementedError

    @property
    def train_set(self):
        pass

    @property
    def valid_set(self):
        pass

    @property
    def test_set(self):
        pass

    @staticmethod
    def one_hot(targets, n_classes):
        pass


class MNIST(Dataset):

    def __init__(self, data_dir, one_hot=True):
        self._urls = ("https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz",)
        self._checksums = ("98100ca27dc0e07ddd9f822cf9d244db",)
        self._n_classes = 10
        super().__init__(data_dir, one_hot=one_hot)

    def _parse(self, **kwargs):
        pass


class FashionMNIST(Dataset):

    def __init__(self, data_dir, one_hot=True):
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        self._urls = [base_url + "train-images-idx3-ubyte.gz",
                      base_url + "train-labels-idx1-ubyte.gz",
                      base_url + "t10k-images-idx3-ubyte.gz",
                      base_url + "t10k-labels-idx1-ubyte.gz"]
        self._checksums = ["8d4fb7e6c68d591d4c3dfef9ec88bf0d",
                           "25c81989df183df01b3e8a0aad5dffbe",
                           "bef4ecab320f06d8554ea6380940ec79",
                           "bb300cfdad3c16e7a12a480ee83cd310"]
        self._n_classes = 10
        super().__init__(data_dir, one_hot=one_hot)

    @staticmethod
    def read_idx(filename):
        pass

    def _parse(self, **kwargs):
        pass


class Cifar(Dataset):

    @staticmethod
    def _cifar_normalize(data):
        pass

    def _parse(self, **kwargs):
        raise NotImplementedError

    def _parse_tarfile(self):
        pass


class Cifar10(Cifar):

    def __init__(self, data_dir, one_hot=False, normalize=False):
        self._urls = ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",)
        self._checksums = ("c58f30108f718f92721af3b95e74349a",)
        self._n_classes = 10
        super().__init__(data_dir, one_hot=one_hot, normalize=normalize)

    def _parse(self, **kwargs):
        pass


class Cifar100(Cifar):

    def __init__(self, data_dir, one_hot=False, normalize=False):
        self._urls = ("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",)
        self._checksums = ("eb9058c3a382ffc7106e4002c42a8d85",)
        self._n_classes = 100
        super().__init__(data_dir, one_hot=one_hot, normalize=normalize)

    def _parse(self, **kwargs):
        pass
