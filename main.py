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


transform_train = transforms.Compose([
                                        transforms.ToTensor(),
                                    ])

path2data = 'C:/Users/Clown1320/PycharmProjects/pytorch_object_detection/Data/voc'
# train_12_ds = dataload_voc(path2data, year = '2012', image_set = 'train', download = False, transform = transform_train)
# train_07_ds = dataload_voc(path2data, year = '2007', image_set = 'train', download = False, transform = transform_train)
train_12_ds = dataload_voc(path2data, year = '2012', image_set = 'train', download = False)
train_07_ds = dataload_voc(path2data, year = '2007', image_set = 'train', download = False)
trainset = DataLoader(train_12_ds+train_07_ds, 1, shuffle = True)



#
# def custom_imshow(img):
#     img = img.numpy()
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.show()


# transforms 정의
IMAGE_SIZE = 512
scale = 1.0

#
# # transforms 적용하기
# trainset.transforms = train_transforms


#


s