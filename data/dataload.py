
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
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import yaml
from dataset.pascalvoc import PascalVoc
from utill.utills import DataEncoder
import cv2

with open('../data/voc.yaml') as file:
    voc_data = yaml.load(file, Loader=yaml.FullLoader)

path2data = '../Data/voc'
voc_class = voc_data['class']


class VOCDataset(VOCDetection):
    def __init__(self, cfg_yaml, root: str):
        super().__init__(root)
        with open(cfg_yaml) as file:
            voc_data = yaml.load(file, Loader=yaml.FullLoader)

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert('RGB'))
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())  # xml파일 분석하여 dict으로 받아오기

        targets = []  # 바운딩 박스 좌표
        labels = []  # 바운딩 박스 클래스

        # 바운딩 박스 정보 받아오기
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'],  t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], voc_class.index(t['name'])
            targets.append(list(label[:4]))  # 바운딩 박스 좌표
            labels.append(label[4])  # 바운딩 박스 클래스

        # if self.transforms:
        #     augmentations = self.transforms(image = img, bboxes = targets)
        #     img = augmentations['image']
        #     targets = augmentations['bboxes']
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return {'img': image, 'targets': targets, 'lables': labels}

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, any]:

        """
        :param node:
        :return:
        """
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


def data_set_show(img, targets, labels, classes):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    targets = np.array(targets)
    W, H = img.size
    colors = np.random.randint(0, 255, size = (80, 3), dtype = 'uint8')  # 바운딩 박스 색상

    for tg, label in zip(targets, labels):
        id_ = int(label)  # class
        bbox = tg[:4]  # [x1, y1, x2, y2]
        color = [int(c) for c in colors[id_]]
        name = classes[id_]
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline = tuple(color), width = 3)
        draw.text((bbox[0], bbox[1]), name, fill = (0, 0, 0, 0))

    plt.imshow(np.array(img))
    plt.show()


if __name__ == '__main__':
    def voc_collect(samples):
        imgs = [sample['img'] for sample in samples]
        targets = [sample['targets'] for sample in samples]
        lables = [sample['lables'] for sample in samples]
        padded_imgs = torch.nn.utils.rnn.pad_sequence(imgs, batch_first=True)
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
        padded_tlables = torch.nn.utils.rnn.pad_sequence(lables, batch_first=True)
        return padded_imgs, padded_targets, padded_tlables

    # transforms 적용하기
    # transforms = train_transforms
    #
    # transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    with open('./voc.yaml') as file:
        voc_data = yaml.load(file, Loader = yaml.FullLoader)
    # train_12_ds = dataload_voc(voc_data['train'], year = '2012', image_set = 'train', download = False)
    # train_07_ds = dataload_voc(voc_data['train'], year = '2007', image_set = 'train', download = False)
    # print(f'12 {len(train_12_ds)}, 07 {len(train_07_ds)}')
    # train_ds = train_12_ds + train_07_ds
    a = PascalVoc(root = "../data/voc/", year = "2007", image_set = "train", download = False, transforms = data_transform)
    b = PascalVoc(root = "../data/voc/", year = "2012", image_set = "train", download = False, transforms = data_transform)
    train_ds = a + b


    test_data = DataLoader(train_ds, batch_size = 20, shuffle = True, num_workers = 4, collate_fn=voc_collect)

    # for i in range(10):
    #     img, target, label = train_ds[i]
    #     print(f'{img.size()=}\n {target.size()=} \n {label.size()=}')
        # plt.figure(figsize = (10, 10))
        # data_set_show(img, target, label, voc_data['class'])
    for i, (img, targets, labels) in enumerate(test_data):
        print(img.shape)
        print(targets.shape)
        print(labels.shape)
    #     d = targets
    #     e = labels
    # for data in test_data:
    #     print(data['img'])
# import torch
# import xml.etree.ElementTree as et
# import os
# import cv2
# import numpy as np
# from torchvision import transforms
# from PIL import Image
# import random
#
#
#
# def flip(img, boxes):
#     img = img.transpose(Image.FLIP_LEFT_RIGHT)
#     w = img.width
#     if boxes.shape[0] != 0:
#         boxes[:, 2] = w - boxes[:,2]
#         boxes[:, 0] = w - boxes[:,0]
#     return img, boxes
#
#
# class VOCDataset(torch.utils.datasets):
#     CLASSES_NAME = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#                     "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor",)
#
#     def __init__(self,root_dir, resize_size=[800, 1333], split='trainval',use_difficult=False, is_train=True, augment=None):
#
#         self.root = root_dir
#         self.use_difficult = use_difficult
#         self.imgset = split
#         self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
#         self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
#         self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
#
#         with open(self._imgsetpath % self.imgset) as f:
#             self.img_ids = f.readlines()
#         self.img_ids = [x.strip() for x in self.img_ids]
#         self.name2id = dict(zip(VOCDataset.CLASSES_NAME,range(len(VOCDataset.CLASSES_NAME))))
#         self.id2name = {v:k for k,v in self.name2id.items()}
#         self.resize_size = resize_size
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]
#         self.train = is_train
#         self.augment = augment
#         print("INFO=====>voc dataset init finished  ! !")
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def __getitem__(self,index):
#
#         img_id = self.img_ids[index]
#         img = Image.open(self._imgpath % img_id)
#
#         anno = et.parse(self._annopath % img_id).getroot()
#         boxes = []
#         classes = []
#         for obj in anno.iter("object"):
#             difficult = int(obj.find("difficult").text) == 1
#             if not self.use_difficult and difficult:
#                 continue
#             _box = obj.find("bndbox")
#             # Make pixel indexes 0-based
#             # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
#             box = [
#                 _box.find("xmin").text,
#                 _box.find("ymin").text,
#                 _box.find("xmax").text,
#                 _box.find("ymax").text,]
#             TO_REMOVE = 1
#             box = tuple(map(lambda x: x - TO_REMOVE, list(map(float, box))))
#             boxes.append(box)
#             name = obj.find("name").text.lower().strip()
#             classes.append(self.name2id[name])
#
#         boxes = np.array(boxes,dtype=np.float32)
#         if self.train:
#             if random.random() < 0.5:
#                 img, boxes = flip(img, boxes)
#             if self.augment is not None:
#                 img, boxes = self.augment(img, boxes)
#         img = np.array(img)
#         img,boxes = self.preprocess_img_boxes(img,boxes,self.resize_size)
#
#         img=transforms.ToTensor()(img)
#         boxes=torch.from_numpy(boxes)
#         classes=torch.LongTensor(classes)
#
#         return img,boxes,classes
#
#
#     def preprocess_img_boxes(self, image, boxes, input_ksize):
#
#         '''
#         resize image and bboxes
#         Returns
#         image_paded: input_ksize
#         bboxes: [None,4]
#         '''
#
#         min_side, max_side = input_ksize
#         h,  w, _ = image.shape
#
#         smallest_side = min(w,h)
#         largest_side = max(w,h)
#         scale = min_side / smallest_side
#         if largest_side * scale > max_side:
#             scale = max_side / largest_side
#         nw, nh = int(scale * w), int(scale * h)
#         image_resized = cv2.resize(image, (nw, nh))
#
#         pad_w = 32-nw % 32
#         pad_h = 32-nh % 32
#
#         image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
#         image_paded[:nh, :nw, :] = image_resized
#
#         if boxes is None:
#             return image_paded
#
#         else:
#             boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
#             boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
#             return image_paded, boxes
#
#     def collate_fn(self,data):
#         imgs_list,boxes_list,classes_list = zip(*data)
#         assert len(imgs_list) == len(boxes_list) == len(classes_list)
#         batch_size = len(boxes_list)
#         pad_imgs_list = []
#         pad_boxes_list = []
#         pad_classes_list = []
#
#         h_list = [int(s.shape[1]) for s in imgs_list]
#         w_list = [int(s.shape[2]) for s in imgs_list]
#         max_h = np.array(h_list).max()
#         max_w = np.array(w_list).max()
#
#         for i in range(batch_size):
#             img = imgs_list[i]
#             pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))
#
#
#         max_num = 0
#         for i in range(batch_size):
#             n = boxes_list[i].shape[0]
#             if n > max_num : max_num = n
#         for i in range(batch_size):
#             pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
#             pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
#
#         batch_boxes = torch.stack(pad_boxes_list)
#         batch_classes = torch.stack(pad_classes_list)
#         batch_imgs = torch.stack(pad_imgs_list)
#
#         return batch_imgs,batch_boxes,batch_classes
#
#
# if __name__ == "__main__":
#     pass
#     eval_dataset = VOCDataset(root_dir='/Users/VOCdevkit/VOCdevkit/VOC0712', resize_size=[800, 1333],
#                                split='test', use_difficult=False, is_train=False, augment=None)
#     print(len(eval_dataset.CLASSES_NAME))
#     #dataset=VOCDataset("/home/data/voc2007_2012/VOCdevkit/VOC2012",split='trainval')
#     # for i in range(100):
#     #     img,boxes,classes=dataset[i]
#     #     img,boxes,classes=img.numpy().astype(np.uint8),boxes.numpy(),classes.numpy()
#     #     img=np.transpose(img,(1,2,0))
#     #     print(img.shape)
#     #     print(boxes)
#     #     print(classes)
#     #     for box in boxes:
#     #         pt1=(int(box[0]),int(box[1]))
#     #         pt2=(int(box[2]),int(box[3]))
#     #         img=cv2.rectangle(img,pt1,pt2,[0,255,0],3)
#     #     cv2.imshow("test",img)
#     #     if cv2.waitKey(0)==27:
#     #         break
#     #imgs,boxes,classes=eval_dataset.collate_fn([dataset[105],dataset[101],dataset[200]])
#     # print(boxes,classes,"\n",imgs.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,imgs.dtype)
#     # for index,i in enumerate(imgs):
#     #     i=i.numpy().astype(np.uint8)
#     #     i=np.transpose(i,(1,2,0))
#     #     i=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
#     #     print(i.shape,type(i))
#     #     cv2.imwrite(str(index)+".jpg",i)
#
#





