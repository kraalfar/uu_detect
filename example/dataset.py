import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms


class PUDataset:
    def __init__(self, data, c=0.5):
        self.data = data
        self.c = c
        self.s = [int(np.random.rand() > 0.5) if i in [0, 1, 8, 9] else 0 for i in data.targets]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y, self.s[i]


def get_data():
    train = torchvision.datasets.CIFAR10("./data", train=True,
                                         transform=transforms.ToTensor(),
                                         download=True)

    test = torchvision.datasets.CIFAR10("./data", train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)

    traindata = PUDataset(train)
    testdata = PUDataset(test)

    return traindata, testdata
