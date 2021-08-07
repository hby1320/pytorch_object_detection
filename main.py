import torch
import torchvision
import tqdm
from torch.nn import *
import torchsummary as summary
from torchvision.datasets import VOCDetection, CocoDetection, ImageFolder
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont  # image processing
import albumentations as a  # Data augmentation
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from PIL import Image
from util.dataload import dataload_voc
from efficientnet_pytorch import EfficientNet #pip install efficientnet_pytorch
from torchsummary import summary
from tqdm import tqdm

input_size = 512
batch = 1
local_rank = -1
path2data = './Data/voc'
backbone = 'efficientnet-b0'
# train_set = []

# for i in range(2):
#     yr = ['2007','2012'] # 2007 : 2501
#     Set = ['train', 'tes', 'val']
#     print(f'{yr[i]}')
#     data = dataload_voc(path2data, year = yr[i], image_set = 'train', download = False)
#     train_set += data
#     print(f'train_set:{len(train_set)}')




# train_12_ds = dataload_voc(path2data, year = '2012', image_set = 'train', download = False)
# train_07_ds = dataload_voc(path2data, year = '2007', image_set = 'train', download = False)
# train_set = DataLoader(train_12_ds+train_07_ds, 1, shuffle = True) # <class 'torch.utils.data.dataloader.DataLoader'>
# transform_train = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
