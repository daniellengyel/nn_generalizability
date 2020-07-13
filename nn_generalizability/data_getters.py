import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import os, pickle

from .save_load import *


PATH_TO_DATA = "{}/data".format(os.environ["PATH_TO_GEN_FOLDER"])

def get_data(data_name, vectorized=False, reduce_train_per=None, seed=0):
    if data_name == "gaussian":
        train_data, test_data = _get_gaussian(seed=seed)
    elif data_name == "mis_gauss":
        train_data, test_data = _get_mis_gauss(seed=seed)
    elif data_name == "MNIST":
        train_data, test_data = _get_MNIST()
    elif data_name == "CIFAR10":
        train_data, test_data = _get_CIFAR10()
    elif data_name == "FashionMNIST":
        train_data, test_data = _get_FashionMNIST()
    else:
        raise NotImplementedError("{} is not implemented.".format(data_name))
    if reduce_train_per is not None:
        train_data = ReducedData(train_data, reduce_train_per, seed=seed)
    if vectorized:
        train_data, test_data = VectorizedWrapper(train_data), VectorizedWrapper(test_data)
    return train_data, test_data


def get_random_data_subset(data, num_datapoints=1, seed=0):
    set_seed(seed)
    data_loader = DataLoader(data, batch_size=num_datapoints, shuffle=True)
    return next(iter(data_loader))


def _get_MNIST():
    train_data = torchvision.datasets.MNIST(os.path.join(PATH_TO_DATA, "MNIST"), train=True,
                                            download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                            ]))
    test_data = torchvision.datasets.MNIST(os.path.join(PATH_TO_DATA, "MNIST"), train=False,
                                           download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))

    return train_data, test_data


def _get_FashionMNIST():
    train_data = torchvision.datasets.FashionMNIST(
        root=os.path.join(PATH_TO_DATA, "FashionMNIST"),
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    test_data = torchvision.datasets.FashionMNIST(
        root=os.path.join(PATH_TO_DATA, "FashionMNIST"),
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    return train_data, test_data

def _get_CIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.CIFAR10(root=os.path.join(PATH_TO_DATA, "CIFAR10"), train=True,
                                            download=True, transform=transform)

    test_data = torchvision.datasets.CIFAR10(root=os.path.join(PATH_TO_DATA, "CIFAR10"), train=False,
                                           download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_data, test_data

def _get_gaussian(seed=0):
    set_seed(seed)
    # get data
    gaussian_params = []

    cov_1 = np.array([[1, 1 / 2.], [1 / 2., 1]])
    cov_2 = np.array([[1, 1 / 2.], [1 / 2., 1]])

    mean_1 = np.array([0, 0])
    mean_2 = np.array([2, 0])

    means = [mean_1, mean_2]
    covs = [cov_1, cov_2]
    training_nums = 500
    test_nums = 100

    train_gaussian = GaussianMixture(means, covs, len(means) * [training_nums])
    test_gaussian = GaussianMixture(means, covs, len(means) * [test_nums])

    return train_gaussian, test_gaussian

def _get_mis_gauss(seed=0):
    set_seed = seed
    # get data
    gaussian_params = []

    cov_1 = np.array([[1, 0], [0, 1]])

    mean_1 = np.array([0, 0])

    means = [mean_1]
    covs = [cov_1]
    training_nums = 200
    test_nums = 120000

    train_gaussian = MisGauss(means, covs, len(means) * [training_nums])
    test_gaussian = MisGauss(means, covs, len(means) * [test_nums])

    return train_gaussian, test_gaussian

class GaussianMixture(Dataset):
    """Dataset gaussian mixture. Points of first gaussian are mapped to 0 while points in the second are mapped 1.

    Parameters
    ----------
    means:
        i: mean
    covs:
        i: cov
    nums:
        i: num for ith class
    """

    def __init__(self, means, covs, nums):

        self.data = []
        self.targets = []

        self.num_classes = len(covs)

        xs = None
        ys = []
        for i in range(len(covs)):
            mean = means[i]
            cov = covs[i]
            num = nums[i]
            x = np.random.multivariate_normal(mean, cov, num)
            if xs is None:
                xs = x
            else:
                xs = np.concatenate([xs, x], axis=0)
            ys += num * [i]

        self.data = torch.Tensor(xs)

        targets = np.array(ys)  # np.eye(self.num_classes)[ys]
        self.targets = torch.Tensor(targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index].long()

    def __len__(self):
        return len(self.data)


class MisGauss(Dataset):
    """Dataset gaussian mixture. Points of first gaussian are mapped to 0 while points in the second are mapped 1.

    Parameters
    ----------
    means:
        i: mean
    covs:
        i: cov
    nums:
        i: num for ith class
    """

    def __init__(self, means, covs, nums):
        self.data = []
        self.targets = []

        self.num_classes = len(covs)

        xs = None
        ys = []
        for i in range(len(covs)):
            mean = means[i]
            cov = covs[i]
            num = nums[i]
            x = np.random.multivariate_normal(mean, cov, num)
            y = 1 * (x[:, 0] >= 0)
            y = [self._flip(a, 0.05) for a in y]
            x += np.random.normal(scale=0.15, size=x.shape)

            if xs is None:
                xs = x
            else:
                xs = np.concatenate([xs, x], axis=0)

            ys += y

        self.data = torch.Tensor(xs)

        targets = np.array(ys)  # np.eye(self.num_classes)[ys]
        self.targets = torch.Tensor(targets)

    def _flip(self, val, alpha=0.05):
        should_flip = np.random.uniform(0, 1) < alpha
        if should_flip:
            if val == 0:
                val = 1
            elif val == 1:
                val = 0
        return val

    def __getitem__(self, index):
        return self.data[index], self.targets[index].long()

    def __len__(self):
        return len(self.data)

#  vectorizes data
class VectorizedWrapper():
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        data, target = self.data.__getitem__(item)
        return data.view(-1), target

    def __len__(self):
        return len(self.data)

class ReducedData:
    def __init__(self, data, per, seed=0):
        set_seed(seed)
        self.num_data = len(data)
        self.per = per
        self.idxs = np.random.choice(list(range(self.num_data)), int(self.num_data * self.per))
        self.data = data

    def __getitem__(self, item):
        data, target = self.data.__getitem__(self.idxs[item])
        return data, target

    def __len__(self):
        return int(self.num_data * self.per)
