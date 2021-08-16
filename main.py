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
import time


#
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#             if phase == 'train':
#                 scheduler.step()
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#
#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model
#

# hyper parameter -> YAML fixed


input_size = (512, 512)
batch_size = 1
lr = 0.001
model = ''

local_rank = -1



# lacation
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
