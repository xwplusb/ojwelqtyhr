import torch
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, STL10, FashionMNIST
from torchvision.transforms import ToTensor

class MNIST_E(MNIST):

    def __init__(self, 
                root: str, 
                train: bool = True, 
                transform = None, 
                targets = None, 
                target_transform = None, 
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
                transform = None, 
                targets = None, 
                target_transform = None, 
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
                transform = None, 
                targets = None, 
                target_transform = None, 
                download: bool = False, 
                *args, **kwars) -> None:
        super().__init__(root, train, transform, target_transform, download)

        if targets:
                indices = [torch.where(self.targets == i) for i in targets]
                targets = [self.targets[index] for index in indices]
                data = [self.data[index] for index in indices]
                self.targets = torch.cat(targets, dim=0)
                self.data = torch.cat(data, dim=0)
        print(self.targets.shape)

class STL(STL10):
    def __init__(self, 
                root: str, 
                split: str = 'train', 
                transform = None, 
                targets = None, 
                target_transform = None, 
                download: bool = False, 
                *args, **kwars) -> None:
        super().__init__(root, split, transform=transform,target_transform=target_transform, download=download)

        if targets:
                # self.labels = torch.from_numpy(self.labels)
                # self.data = torch.from_numpy(self.data)
                
                indices = [np.where(self.labels == i) for i in targets]
                targets = [self.labels[index] for index in indices]
                data = [self.data[index] for index in indices]
                self.labels = np.concatenate(targets)
                self.data = np.concatenate(data)
