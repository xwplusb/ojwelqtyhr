from typing import Any, Callable, Optional
import torch
import numpy as np

from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10, STL10, FashionMNIST
from torchvision.datasets import ImageFolder


class MNIST_E(MNIST):

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform=None,
                 targets=None,
                 target_transform=None,
                 download: bool = False,
                 *args, **kwars) -> None:
        super().__init__(root, train, transform, target_transform, download)
        if targets:
            indices = [torch.where(self.targets == i) for i in targets]
            targets = [self.targets[index] for index in indices]
            data = [self.data[index] for index in indices]
            self.targets = torch.cat(targets, dim=0)
            self.data = torch.cat(data, dim=0)


class FashionMNIST_E(FashionMNIST):

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform=None,
                 targets=None,
                 target_transform=None,
                 download: bool = True,
                 *args, **kwars) -> None:
        super().__init__(root, train, transform, target_transform, download)

        if targets:
            indices = [torch.where(self.targets == i) for i in targets]
            targets = [self.targets[index] for index in indices]
            data = [self.data[index] for index in indices]
            self.targets = torch.cat(targets, dim=0)
            self.data = torch.cat(data, dim=0)


class CIFAR(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform=None,
                 targets=None,
                 target_transform=None,
                 download: bool = False,
                 *args, **kwars) -> None:
        super().__init__(root, train, transform, target_transform, download)


        if targets:
            self.data = list(self.data[i] for i in range(len(self.targets)) if self.targets[i] in targets)
            self.targets = list(target for target in self.targets if target in targets)


class STL(STL10):
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform=None,
                 targets=None,
                 target_transform=None,
                 download: bool = False,
                 *args, **kwars) -> None:
        super().__init__(root, split, transform=transform, target_transform=target_transform, download=download)

        if targets:
            # self.labels = torch.from_numpy(self.labels)
            # self.data = torch.from_numpy(self.data)

            indices = [np.where(self.labels == i) for i in targets]
            targets = [self.labels[index] for index in indices]
            data = [self.data[index] for index in indices]
            self.labels = np.concatenate(targets)
            self.data = np.concatenate(data)


class GTRSB(ImageFolder):

    def __init__(self, root: str, transform, targets, *args, **kwars):
        super().__init__(root, transform)
        _targets = [1, 2, 4, 9, 11, 18, 25, 26, 35, 36, 37, 38]
        self.re_index = dict( zip(_targets, range(len(_targets))))

        self.samples = list( tuple((samp[0], self.re_index[samp[1]])) for samp in self.samples if samp[1] in _targets)
        self.targets = list( samp[1] for samp in self.samples )

        if targets:
            self.samples = list( samp for samp in self.samples if samp[1] in targets)
            self.targets = list(samp[1] for samp in self.samples)


class SVHN(Dataset):

    def __init__(self, root, transform):
        self.data = loadmat(root)
        self.targets = self.data['y']
        self.data = self.data['X']

        if transform:
            self.trans = transform
    def __getitem__(self, item):
        data, target = self.data[item], self.targets[item]
        if self.trans:
            data = self.trans(data)
        return data, target



