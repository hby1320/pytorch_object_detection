from typing import Optional, Callable
import torch
import torchvision
from torch.utils.data import DataLoader, ChainDataset, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np


class PascalVoc(torchvision.datasets.VOCDetection):
    def __init__(self,
                 root: str,
                 year: str = "2012",
                 image_set: str = "train",
                 download: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,):

        super(PascalVoc, self).__init__(root, year, image_set, download,  transforms, transform, target_transform)



# if __name__ == '__main__':
#     transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
#     a = PascalVoc(root = "../data/voc", year="2007", image_set="train", download=False, transform = transform_train)
#     b = PascalVoc(root = "../data/voc", year="2012", image_set = "train", download = False, transform = transform_train)
#     c = ConcatDataset([a, b])
#     test_data = DataLoader(c, batch_size = 1, shuffle = True, num_workers = 4)
#     for i, data in enumerate(test_data):
#         print(f'{data}')


