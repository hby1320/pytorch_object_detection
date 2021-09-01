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
        super().__init__(root, year, image_set, download, transforms)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())  # xml파일 분석하여 dict으로 받아오기

        targets = []  # 바운딩 박스 좌표
        labels = []  # 바운딩 박스 클래스
        voc_class = ["aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        # 바운딩 박스 정보 받아오기

        for t in target['annotation']['object']:
            label = np.zeros(5)  # lable (xmin, ymin, xmax, ymax, class name)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'],  t['bndbox']['ymax'], \
                       voc_class.index(t['name'])
            targets.append(list(label[:4]))  # 바운딩 박스 좌표
            labels.append(label[4])  # 바운딩 박스 클래스
        sample = {'img': img, 'target': targets}

        # if self.transforms:
        #     img, target = self.transforms(img, target)
        #
        #     # augmentations = self.transforms(transform = img, target_transform = targets)
        #     # img = augmentations['image']
        #     # targets = augmentations['bboxes']
        # # if self.transforms is not None:
        # #     img, target = self.transforms(img, target)
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)
        if self.transforms:
            img, target = self.transforms(sample)
            # img = augmentations['image']
            # targets = augmentations['bboxes']

        return img, target, labels

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

# def retina_collate_fn(batch):
#     encoder = DataEncoder()
#     imgs = [x[0] for x in batch]
#     boxes = [torch.Tensor(x[1]) for x in batch]
#     labels = [torch.Tensor(x[2]) for x in batch]
#     h, w = 600, 600
#     num_imgs = len(imgs)
#     inputs = torch.zeros(num_imgs, 3, h, w)
#
#     loc_targets = []
#     cls_targets = []
#     for i in range(num_imgs):
#         inputs[i] = imgs[i]
#         loc_target, cls_target = encoder.encode(boxes = boxes[i], labels = labels[i], input_size = (w, h))
#         loc_targets.append(loc_target)
#         cls_targets.append(cls_target)
#     return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

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
# def collate_fn(batch):
#     imgs = [torch.Tensor(x[0]) for x in batch]
#     boxes = [torch.Tensor(x[1]) for x in batch]
#     labels = [torch.Tensor(x[2]) for x in batch]
#     print(imgs,labels,boxes)
#     # imgs, targets, labels = zip(*batch)
#     # batch_size = len(imgs)
#     # imgs_list = []
#     # targets_list = []
#     # labels_list = []
#
#
#     # imgs = torch.from_numpy(np.stack(imgs, axis=0))
#     #
#     # max_num_annots = max(target.shape[0] for target in targets)
#     #
#     # if max_num_annots > 0:
#     #
#     #     annot_padded = torch.ones((len(targets), max_num_annots, 5)) * -1
#     #
#     #     for idx, target in enumerate(targets):
#     #         if target.shape[0] > 0:
#     #             annot_padded[idx, :target.shape[0], :] = target
#     # else:
#     #     annot_padded = torch.ones((len(targets), 1, 5)) * -1
#     #
#     # # imgs = imgs.permute(0, 3, 1, 2)
#
#     return imgs, targets, labels

#
# def collate_fn(batch):
#     images = list()
#     boxes = list()
#     labels = list()
#
#     for b in batch:
#         images.append(b[0])
#         boxes.append(b[1])
#         labels.append(b[2])
#     torch.tensor(images)
#     images = torch.stack(images, dim = 0)
#
#     return images, boxes, labels
# def collate_fn(batch):
#     imgs_list, boxes_list, classes_list = zip(*batch)
#
#     assert len(imgs_list) == len(boxes_list) == len(classes_list)
#     batch_size = len(boxes_list)
#     pad_imgs_list = []
#     pad_boxes_list = []
#     pad_classes_list = []
#     #imgs_list = torch.tensor(imgs_list)
#
#     # imgs_list = imgs_list.permute(0, 3, 1, 2)
#
#
#     h_list = [int(s.shape[1]) for s in imgs_list]
#     w_list = [int(s.shape[2]) for s in imgs_list]
#     max_h = torch.max(torch.tensor(h_list))
#     max_w = torch.max(torch.tensor(w_list))
#
#     for i in range(batch_size):
#         img = torch.tensor(imgs_list[i])
#         # img = img.permute(3, 1, 2)
#         pad_imgs_list.append(torch.nn.functional.pad(img,(0, int(max_w - img.shape[1]), 0, int(max_h - img.shape[0])),
#                                                      value=0.))
#
#         print(max_h, max_w)
#         print(pad_imgs_list)
#     max_num = 0
#     for i in range(batch_size):
#         n = torch.tensor(boxes_list[i]).shape[0]
#         if n > max_num:
#             max_num = n
#     for i in range(batch_size):
#         boxes = torch.tensor(boxes_list[i])
#         classes = torch.tensor(classes_list[i])
#         pad_boxes_list.append(torch.nn.functional.pad(boxes, (0, 0, 0, max_num - boxes.shape[0]), value = -1))
#         pad_classes_list.append(torch.nn.functional.pad(classes, (0, max_num - classes.shape[0]), value = -1))
#     batch_boxes = torch.stack(pad_boxes_list)
#     batch_classes = torch.stack(pad_classes_list)

#     batch_imgs = torch.stack(pad_imgs_list)
#     # batch_boxes = batch_boxes.permute(0, 3, 1, 2)
#     # batch_classes = batch_classes.permute(0, 3, 1, 2)
#     batch_imgs = batch_imgs.permute(0, 3, 1, 2)
#     return batch_imgs, batch_boxes, batch_classes

#
# def collate_fn(batch):
#     imgs_list, boxes_list, classes_list = zip(*batch)
#     assert len(imgs_list) == len(boxes_list) == len(classes_list)
#     batch_size = len(boxes_list)
#     out_imgs_list = []
#     out_boxes_list = []
#     out_classes_list = []
#     for i in range(batch_size):
#         print(imgs_list[i])
#         print(i, batch_size)
#         img = imgs_list[i]
#         # img = img.permute(3, 1, 2)
#         out_imgs_list.append(img)
#     max_num = 0
#     for i in range(batch_size):
#         n = torch.tensor(boxes_list[i]).shape[0]
#         if n > max_num:
#             max_num = n
#     for i in range(batch_size):
#         boxes = torch.tensor(boxes_list[i])
#         classes = torch.tensor(classes_list[i])
#         out_boxes_list.append(torch.nn.functional.pad(boxes, (0, 0, 0, max_num - boxes.shape[0]), value = -1))
#         out_classes_list.append(torch.nn.functional.pad(classes, (0, max_num - classes.shape[0]), value = -1))
#     batch_boxes = torch.stack(out_boxes_list)
#     batch_classes = torch.stack(out_classes_list)
#     batch_imgs = torch.stack(out_imgs_list)
#     batch_imgs = batch_imgs.permute(0, 3, 1, 2)
#     return batch_imgs, batch_boxes, batch_classes
def collect_fn(self, data):
    img = [s['img'] for s in data]



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(512)])
    # transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    a = PascalVoc(root = "../data/voc", year = "2007", image_set = "train", download = False, )
    # b = PascalVoc(root = "../data/voc", year = "2012", image_set = "train", download = False)
    # c = ConcatDataset([a, b])
    # test_data = DataLoader(a, batch_size = 1, shuffle = True, num_workers = 4)
    # #
    # # bar = enumerate(test_data)
    # # for (img, targets, labels) in test_data:
    # #     print(labels)
    # train_datalodaer = DataLoader(dataset = c, batch_size = 1, shuffle = True, num_workers=4, pin_memory = True)
    #
    # for batch_idx, (imgs, targets, classes) in enumerate(train_datalodaer):
    #     print(type(imgs))

    # for i in range(5):
    #     img, box, cls = a[i]
    #     tf = transforms.ToTensor()
        # tf = transforms.Compose([transforms.ColorJitter(brightness=0.5,saturation=0,contrast=0,hue=(0)),
        #                          transforms.ToTensor(),])
        # print(f'{type(img)}')
        # img = tf(img)
        # print(f'{type(img)}')
        # box = torch.tensor(box)
        # img = img.permute(1, 2, 0)
        # plt.imshow(img)
        # plt.show()
        # print(f'{img.shape= }')  # img.shape= torch.Size([3, 333, 500])
        # print(f'{box.shape= }')  # box.shape= torch.Size([1, 4])  #
        # print(f'{cls= }')  # cls= [6.0]


    def visualize_agm(path='./001', ):
        amg = ['brightness', 'contrast', 'saturation', 'RandomAdjustSharpness', 'RandomCrop', 'RandomHorizontalFlip']
        acept = []
        for i in range(len(amg)):
            print(f'agment List {i+1}: {amg[i]}')
            acept = acept.append(input(f'input 1 else 0'))

        img = Image.open(f'{path}.png')
        tf = transforms.ToTensor()
        img = tf(img)

        amgent = []
        if acept[0] ==1:
            amgent.append(transforms.ColorJitter((5,5), 0, 0, 0))
        elif acept[1]==1:
            amgent.append(transforms.ColorJitter(0, (5,5), 0, 0))
        elif acept[2]==1:
            amgent.append(transforms.ColorJitter(0, 0, (5,5), 0))
        elif acept[3]==1:
            amgent.append(transforms.RandomAdjustSharpness(1))
        elif acept[4]==1:
            amgent.append(transforms.RandomCrop((512,1024)))
        elif acept[5]==1:
            amgent.append(transforms.RandomHorizontalFlip(1))
        img_list = []
        print(f'{amgent}')

    visualize_agm()
    #
    # img = Image.open(f'./001.png')
    #
    # tf = transforms.ToTensor()
    # tf1 = transforms.ColorJitter((5,5), 0, 0, 0)
    # tf2 = transforms.ColorJitter(0, (5,5), 0, 0)
    # tf3 = transforms.ColorJitter(0, 0, (5,5), 0)
    # tf4 = transforms.RandomAdjustSharpness(1)
    # tf5 = transforms.RandomCrop((512,1024))
    # tf6 = transforms.RandomHorizontalFlip(1)
    # img = tf(img)
    # img1 = tf1(img)
    # img2 = tf2(img)
    # img3 = tf3(img)
    # img4 = tf4(img)
    # img5 = tf5(img)
    # img6 = tf6(img)
    #
    #
    # # print(f'{img.size()}')
    # # print(f'{type(img)}')
    # img1 = img1.permute(1, 2, 0)
    # img2 = img2.permute(1, 2, 0)
    # img3 = img3.permute(1, 2, 0)
    # img4 = img4.permute(1, 2, 0)
    # img5 = img5.permute(1, 2, 0)
    # img6 = img6.permute(1, 2, 0)
    #
    # # plt.subplot(3, 3, 1)
    # plt = plt.figure(figsize = (75,75), dpi = 300)
    # ax1 = plt.add_subplot(3, 3, 1)
    # ax2 = plt.add_subplot(3, 3, 2)
    # ax3 = plt.add_subplot(3, 3, 3)
    # ax4 = plt.add_subplot(1, 3, 1)
    # ax5 = plt.add_subplot(1, 3, 2)
    # ax6 = plt.add_subplot(1, 3, 3)
    #
    # ax1.set_title(f'brightness',fontsize=6)
    # ax2.set_title(f'contrast',fontsize=6)
    # ax3.set_title(f'saturation',fontsize=6)
    # ax4.set_title(f'RandomAdjustSharpness',fontsize=6)
    # ax5.set_title(f'RandomCrop',fontsize=6)
    # ax6.set_title(f'RandomHorizontalFlip',fontsize=6)
    # ax1.axis('off')
    # ax2.axis('off')
    # ax3.axis('off')
    # ax4.axis('off')
    # ax5.axis('off')
    # ax6.axis('off')
    # ax1.set_xticks([]), ax1.set_yticks([])
    # ax2.set_xticks([]), ax2.set_yticks([])
    # ax3.set_xticks([]), ax3.set_yticks([])
    # ax4.set_xticks([]), ax4.set_yticks([])
    # ax5.set_xticks([]), ax5.set_yticks([])
    # ax6.set_xticks([]), ax6.set_yticks([])
    #
    # ax1.imshow(img1)
    # ax2.imshow(img2)
    # ax3.imshow(img3)
    # ax4.imshow(img4)
    # ax5.imshow(img5)
    # ax6.imshow(img6)
    # plt.title(f'RandomHorizontalFlip')
    # # box = torch.tensor(box)
    # plt.show()
    # print(f'{img.shape= }')  # img.shape= torch.Size([3, 333, 500])
    # # print(f'{box.shape= }')  # box.shape= torch.Size([1, 4])  #
    # # print(f'{cls= }')  # cls= [6.0]
