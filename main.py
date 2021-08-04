import torch
import torchvision
import tqdm
from torch.nn import *
import torchsummary as summary
from torchvision.datasets import VOCDetection, CocoDetection, ImageFolder
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont  # iamgeprossesing
import albumentations as A  # Data agumentation
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






# train_12_ds = dataload_voc(path2data, year = '2012', image_set = 'train', download = False, transform = transform_train)
# train_07_ds = dataload_voc(path2data, year = '2007', image_set = 'train', download = False, transform = transform_train)
train_12_ds = dataload_voc(path2data, year = '2012', image_set = 'train', download = False, transform = transform_train)
train_07_ds = dataload_voc(path2data, year = '2007', image_set = 'train', download = False, transform = transform_train)
trainset = DataLoader(train_12_ds+train_07_ds, 1, shuffle = True) # <class 'torch.utils.data.dataloader.DataLoader'>
transform_train = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


#
# def custom_imshow(img):
#     img = img.numpy()
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.show()




# model.eval()
# for i in range(len(train_07_ds)):
#     img = trainset[i]
#     with torch.no_grad():
#         outputs = model(img)
#
# # transforms 정의
# IMAGE_SIZE = 512
# scale = 1.0
#
# #
# # # transforms 적용하기
# # trainset.transforms = train_transforms


#
