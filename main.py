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
import os
import yaml
import numpy as np
from PIL import Image
from util.dataload import dataload_voc

path2data = '../Data/voc'
train_12_ds = dataload_voc(path2data, year = '2012', image_set = 'train', download = False)
train_07_ds = dataload_voc(path2data, year = '2007', image_set = 'train', download = False)

trainset = DataLoader(train_12_ds+train_07_ds, 1, True)


# transforms 정의
IMAGE_SIZE = 512
scale = 1.0

# 이미지에 padding을 적용하여 종횡비를 유지시키면서 크기가 600x600 되도록 resize 합니다.
train_transforms = A.Compose([
                    A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
                    A.PadIfNeeded(min_height=int(IMAGE_SIZE*scale), min_width=int(IMAGE_SIZE*scale),border_mode=cv2.BORDER_CONSTANT),
                    to_tensor()
                    ],
                    bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4, label_fields=[])
                    )


# transforms 적용하기
trainset.transforms = train_transforms

