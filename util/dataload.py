import torch
import torchvision
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as transform
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, Optional, Tuple, List
import numpy as np
import xml.etree.ElementTree as ET ## voc GT XML
import collections
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import os

## ->FIX 복수의 데이터세트에서 사용가능하게 수정하기  voc폴더에 한번에 불러 올수 있도록 수정 XML 데이터 불러오기 ## YAML 사용해서 파일로 정리
## -> coco 데이터세트도 동일하게 만들기
##
with open('../Data/voc.yaml') as file:
    voc_data = yaml.load(file, Loader=yaml.FullLoader)

path2data = '../Data/voc'
if not os.path.exists(path2data):
    os.mkdir(path2data)


voc_class = voc_data['class']

class dataload_voc(VOCDetection):
    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())  # xml파일 분석하여 dict으로 받아오기

        targets = []  # 바운딩 박스 좌표
        labels = []  # 바운딩 박스 클래스

        # 바운딩 박스 정보 받아오기
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'],  t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], voc_class.index(t['name'])
            targets.append(list(label[:4]))  # 바운딩 박스 좌표
            labels.append(label[4])  # 바운딩 박스 클래스

        if self.transforms:
            augmentations = self.transforms(image = img, bboxes = targets)
            img = augmentations['image']
            targets = augmentations['bboxes']

        return img, targets, labels

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, any]: #parse_voc_xml[str, any]:  # xml 파일을 dictionary로 반환
        voc_dict: Dict[str, any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

def show(img, targets, labels, classes = voc_class):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    targets = np.array(targets)
    W, H = img.size

    for tg, label in zip(targets, labels):
        id_ = int(label)  # class
        bbox = tg[:4]  # [x1, y1, x2, y2]

        color = [int(c) for c in colors[id_]]
        name = classes[id_]

        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline = tuple(color), width = 3)
        draw.text((bbox[0], bbox[1]), name, fill = (255, 255, 255, 0))
    plt.imshow(np.array(img))
    plt.show()






if __name__ == '__main__':
    train_12_ds = dataload_voc(path2data, year = '2012', image_set = 'train', download = False)
    train_07_ds = dataload_voc(path2data, year = '2007', image_set = 'train', download = False)
    print(f'12 {len(train_12_ds)}, 07 {len(train_07_ds)}')
    train_ds = train_12_ds + train_07_ds
    print(f'07+12 {len(train_ds)}')
    img, target, label = train_ds[2]
    colors = np.random.randint(0, 255, size = (80, 3), dtype = 'uint8')  # 바운딩 박스 색상
    plt.figure(figsize = (10, 10))
    show(img, target, label)



# path2data = './data/voc'
# if not os.path.exists(path2data):
#     os.mkdir(path2data)
# voc_class = []
# torchvision.datasets.VOCDetection(path2data,'2012','train',True)
#
#
# #
# # class myVOCDetection(VOCDetection):
# #     def __getitem__(self, index):
# #         img = np.array(Image.open(self.images[index]).convert('RGB'))
# #         target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())  # xml파일 분석하여 dict으로 받아오기
# #
# #         targets = []  # 바운딩 박스 좌표
# #         labels = []  # 바운딩 박스 클래스
# #
# #         # 바운딩 박스 정보 받아오기
# #         for t in target['annotation']['object']:
# #             label = np.zeros(5)
# #             label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox'][
# #                 'ymax'], voc_class.index(t['name'])
# #
# #             targets.append(list(label[:4]))  # 바운딩 박스 좌표
# #             labels.append(label[4])  # 바운딩 박스 클래스
# #
# #         if self.transforms:
# #             augmentations = self.transforms(image=img, bboxes=targets)
# #             img = augmentations['image']
# #             targets = augmentations['bboxes']
# #
# #         return img, targets, labels
# #
# #     def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:  # xml 파일을 dictionary로 반환
# #         voc_dict: Dict[str, Any] = {}
# #         children = list(node)
# #         if children:
# #             def_dic: Dict[str, Any] = collections.defaultdict(list)
# #             for dc in map(self.parse_voc_xml, children):
# #                 for ind, v in dc.items():
# #                     def_dic[ind].append(v)
# #             if node.tag == "annotation":
# #                 def_dic["object"] = [def_dic["object"]]
# #             voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
# #         if node.text:
# #             text = node.text.strip()
# #             if not children:
# #                 voc_dict[node.tag] = text
# #         return voc_dict
#
#
# #resnet50 = torchvision.models.resnet50()
# #print(resnet50)
#
# def device_type(local) -> torch.device:
#     if torch.cuda.is_available():
#         torch.cuda.set_device(local)
#         device_type = torch.device('cuda',local)
#         return device_type
#
#     else:
#         device_type = torch.device('cpu')
#         return device_type
#
#
#
# device = device_type('')
#
# class biled_Dataset(torch.utils.data.Dataset):
#
#     def __init__(self, root, transforms):
#         self.root = root
#         self.transforms = transforms
#         self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
#         self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))
#     # __getitem__ : 로드한 data를 차례차례 돌려줌
#
#     def __getitem__(self, idx):
#         img_path = os.path.join (self.root, "PNGImages", self.imgs[idx])
#         mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
#         img = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path)
#         mask = np.array(mask)
#         obj_ids = np.unique(mask)
#         obj_ids = obj_ids[1:]
#         masks = mask == obj_ids[:, None, None]
#
#         num_objs = len(obj_ids)
#         boxes = []
#         for i in range(num_objs):
#             pos = np.where(masks[i])
#             xmin = np.min(pos[1])
#             xmax = np.max(pos[1])
#             ymin = np.min(pos[0])
#             ymax = np.max(pos[0])
#             boxes.append([xmin, ymin, xmax, ymax])
#
#         boxes = torch.as_tensor(boxes, dtype = torch.float32)
#         labels = torch.ones((num_objs,), dtype = torch.int64)
#         masks = torch.as_tensor(masks, dtype = torch.uint8)
#
#         image_id = torch.tensor([idx])
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype = torch.int64)
#
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd
#
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
#
#         return img, target
#
#     # __len__ : 전체 데이터의 길이를 계산함
#     def __len__(self):
#         return len(self.imgs)
#
#
#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#
# data_dir = 'data/hymenoptera_data'
# image_datasets = {x: ImageFolder(os.path.join(data_dir, x),data_transforms[x])
#                   for x in ['train', 'val']}
#
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
#                                               batch_size=4,
#                                               shuffle=True,
#                                               num_workers=4
#
#                                               )
#               for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = yaml.load_all('./data/voc.yaml', )