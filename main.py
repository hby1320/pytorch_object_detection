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


path2data = './data/voc'
if not os.path.exists(path2data):
    os.mkdir(path2data)
voc_class = []
torchvision.datasets.VOCDetection(path2data,'2012','train',True)


#
# class myVOCDetection(VOCDetection):
#     def __getitem__(self, index):
#         img = np.array(Image.open(self.images[index]).convert('RGB'))
#         target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())  # xml파일 분석하여 dict으로 받아오기
#
#         targets = []  # 바운딩 박스 좌표
#         labels = []  # 바운딩 박스 클래스
#
#         # 바운딩 박스 정보 받아오기
#         for t in target['annotation']['object']:
#             label = np.zeros(5)
#             label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox'][
#                 'ymax'], voc_class.index(t['name'])
#
#             targets.append(list(label[:4]))  # 바운딩 박스 좌표
#             labels.append(label[4])  # 바운딩 박스 클래스
#
#         if self.transforms:
#             augmentations = self.transforms(image=img, bboxes=targets)
#             img = augmentations['image']
#             targets = augmentations['bboxes']
#
#         return img, targets, labels
#
#     def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:  # xml 파일을 dictionary로 반환
#         voc_dict: Dict[str, Any] = {}
#         children = list(node)
#         if children:
#             def_dic: Dict[str, Any] = collections.defaultdict(list)
#             for dc in map(self.parse_voc_xml, children):
#                 for ind, v in dc.items():
#                     def_dic[ind].append(v)
#             if node.tag == "annotation":
#                 def_dic["object"] = [def_dic["object"]]
#             voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
#         if node.text:
#             text = node.text.strip()
#             if not children:
#                 voc_dict[node.tag] = text
#         return voc_dict


#resnet50 = torchvision.models.resnet50()
#print(resnet50)

def device_type(local) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device_type = torch.device('cuda',local)
        return device_type

    else:
        device_type = torch.device('cpu')
        return device_type



device = device_type('')

class biled_Dataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))
    # __getitem__ : 로드한 data를 차례차례 돌려줌

    def __getitem__(self, idx):
        img_path = os.path.join (self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.ones((num_objs,), dtype = torch.int64)
        masks = torch.as_tensor(masks, dtype = torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype = torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # __len__ : 전체 데이터의 길이를 계산함
    def __len__(self):
        return len(self.imgs)



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4

                                              )
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = yaml.load_all('./data/voc.yaml', )
