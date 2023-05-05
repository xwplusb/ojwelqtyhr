from torch.utils.data import DataLoader

from utils.data.dataset import MNIST_E, FashionMNIST_E, CIFAR, STL
from utils.data.transfrom import trans_dict


dataset_dict = {
    'MNIST': MNIST_E,
    'FashionMNIST': FashionMNIST_E,
    'CIFAR': CIFAR,
    'STL': STL
}

def load_data(
        name: str,
        root: str,
        split: str = 'Train',
        train: bool = True, 
        transform = None, 
        targets = None, 
        target_transform = None, 
        download: bool = False,
        *args, **kwars):

    transform = trans_dict[transform]
    data_class = dataset_dict[name]
    data = data_class(root=root, transform=transform, targets=targets, download=download)
    return data