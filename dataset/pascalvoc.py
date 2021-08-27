from typing import Optional, Callable, Dict
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, ChainDataset, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import collections
import tqdm
from utill.utills import DataEncoder
from torchvision import transforms

class PascalVoc(torchvision.datasets.VOCDetection):
    def __init__(self,
                 root: str,
                 year: str = "2012",
                 image_set: str = "train",
                 download: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,):
        super().__init__(root, year, image_set, download,transform, target_transform)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())  # xml파일 분석하여 dict으로 받아오기

        targets = []  # 바운딩 박스 좌표
        labels = []  # 바운딩 박스 클래스
        voc_class = ["aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        # 바운딩 박스 정보 받아오기

        for t in target['annotation']['object']:
            label = np.zeros(5)  # lable(xmin, ymin, xmax, ymax, class name)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'],  t['bndbox']['ymax'], \
                       voc_class.index(t['name'])
            targets.append(list(label[:4]))  # 바운딩 박스 좌표
            labels.append(label[4])  # 바운딩 박스 클래스

        if self.transforms:
            augmentations = self.transforms(image = img, bboxes = targets)
            img = augmentations['image']
            targets = augmentations['bboxes']
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

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

def retina_collate_fn(batch):
    encoder = DataEncoder()
    imgs = [x[0] for x in batch]
    boxes = [torch.Tensor(x[1]) for x in batch]
    labels = [torch.Tensor(x[2]) for x in batch]
    h, w = 600, 600
    num_imgs = len(imgs)
    inputs = torch.zeros(num_imgs, 3, h, w)

    loc_targets = []
    cls_targets = []
    for i in range(num_imgs):
        inputs[i] = imgs[i]
        loc_target, cls_target = encoder.encode(boxes = boxes[i], labels = labels[i], input_size = (w, h))
        loc_targets.append(loc_target)
        cls_targets.append(cls_target)
    return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

# def collate_fn(batch):
#     imgs_list, boxes_list, classes_list = zip(*batch)
#     assert len(imgs_list) == len(boxes_list) == len(classes_list)
#     batch_size = len(boxes_list)
#     imgs_lists = []
#     boxes_lists = []
#     classes_lists = []
#     for i in range(batch_size):
#         img = imgs_list[i]
#         imgs_lists.append(img)
#         boxes = boxes_list[i]
#         boxes_lists.append(boxes)
#         classes = classes_list[i]
#         classes_lists.append(classes)
#
#     batch_boxes = torch.stack(boxes_lists)
#     batch_classes = torch.stack(classes_lists)
#     batch_imgs = torch.stack(imgs_lists)
#
#     return imgs_lists, boxes_lists, classes_lists
def collate_fn(batch):
    imgs = [torch.Tensor(x[0]) for x in batch]
    boxes = [torch.Tensor(x[1]) for x in batch]
    labels = [torch.Tensor(x[2]) for x in batch]
    print(imgs,labels,boxes)
    # imgs, targets, labels = zip(*batch)
    # batch_size = len(imgs)
    # imgs_list = []
    # targets_list = []
    # labels_list = []


    # imgs = torch.from_numpy(np.stack(imgs, axis=0))
    #
    # max_num_annots = max(target.shape[0] for target in targets)
    #
    # if max_num_annots > 0:
    #
    #     annot_padded = torch.ones((len(targets), max_num_annots, 5)) * -1
    #
    #     for idx, target in enumerate(targets):
    #         if target.shape[0] > 0:
    #             annot_padded[idx, :target.shape[0], :] = target
    # else:
    #     annot_padded = torch.ones((len(targets), 1, 5)) * -1
    #
    # # imgs = imgs.permute(0, 3, 1, 2)

    return imgs, targets, labels




# def collate_fn(batch):
    imgs_list, boxes_list, classes_list = zip(*batch)
    assert len(imgs_list) == len(boxes_list) == len(classes_list)
    # batch_size = len(boxes_list)
    # pad_imgs_list = []
    # pad_boxes_list = []
#     pad_classes_list = []
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     h_list = [int(s.shape[1]) for s in imgs_list]
#     w_list = [int(s.shape[2]) for s in imgs_list]
#     max_h = np.array(h_list).max()
#     max_w = np.array(w_list).max()
#     for i in range(batch_size):
#         img = imgs_list[i]
#         pad_imgs_list.append(transforms.Normalize(mean, std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))
#
#     max_num = 0
#     for i in range(batch_size):
#         n = boxes_list[i].shape[0]
#         if n > max_num:
#             max_num = n
#     for i in range(batch_size):
#         pad_boxes_list.append(
#             torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value = -1))
#         pad_classes_list.append(
#             torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value = -1))
#     batch_boxes = torch.stack(pad_boxes_list)
#     batch_classes = torch.stack(pad_classes_list)
#     batch_imgs = torch.stack(pad_imgs_list)
#
#     return batch_imgs, batch_boxes, batch_classes


if __name__ == '__main__':
    # transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    a = PascalVoc(root = "../data/voc", year = "2007", image_set = "train", download = False)
    b = PascalVoc(root = "../data/voc", year = "2012", image_set = "train", download = False)
    c = ConcatDataset([a, b])
    # test_data = DataLoader(c, batch_size = 64, shuffle = True, num_workers = 4)
    #
    # bar = enumerate(test_data)
    # for (img, targets, labels) in test_data:
    #     print(labels)
    train_datalodaer = DataLoader(dataset = c, batch_size = 1, shuffle = True, num_workers=4, pin_memory = True)

    for data in train_datalodaer:
        print(type(data))

