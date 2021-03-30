from dataclasses import dataclass
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, random_split
from typing import NamedTuple

import torchvision.transforms as T

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST.resources = [
    ("/".join([MIRROR, url.split("/")[-1]]), md5)
    for url, md5 in MNIST.resources
]

train_transform = T.Compose([T.RandomRotation(25), T.ToTensor()])
test_transform = T.Compose([T.ToTensor()])


@dataclass
class MNISTDataset:
    train = MNIST(
        root="./dataset", 
        train=True, 
        transform=train_transform, 
        download=True,
    )
    
    test = MNIST(
        root="./dataset", 
        train=False, 
        transform=test_transform, 
        download=True,
    )


@dataclass
class MNISTLoader:
    train = DataLoader(
        MNISTDataset.train,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test = DataLoader(
        MNISTDataset.test,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )