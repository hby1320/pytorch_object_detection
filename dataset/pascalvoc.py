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
                 image_set: str = "trainval",
                 download: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,):
        super().__init__(root, year, image_set, download, transforms)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())  # xml파일 분석하여 dict으로 받아오기

        targets = []  # 바운딩 박스 좌표
        labels = []  # 바운딩 박스 클래스

        voc_class = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                     "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                     "tvmonitor"
                     ]  # Class Name

        # 바운딩 박스 정보 받아오기
        for t in target['annotation']['object']:
            label = np.zeros(5)  # lable (xmin, ymin, xmax, ymax, class name)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'],  t['bndbox']['ymax'], \
                       voc_class.index(t['name'])
            targets.append(label[:4])  # 바운딩 박스 좌표
            labels.append(label[4])  # 바운딩 박스 클래스
        targets = np.array(targets, dtype=np.float32)
        targets = torch.from_numpy(targets)
        labels = torch.LongTensor(labels)
        sample = {'img': image, 'target': targets}

        if self.transforms:
            image, target = self.transforms(sample['img'], sample['target'])

        return {'img': image, 'targets': targets, 'lables': labels}

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image


    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Resize(512)])
    # transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    a = PascalVoc(root = "../data/voc/", year = "2007", image_set = "train", download = False,)
    b = PascalVoc(root = "../data/voc/", year = "2012", image_set = "train", download = False)
    c = PascalVoc(root = "../data/voc/", year = "2007", image_set = "test", download = False)

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


    # def visualize_agm(path='./001', ):
    #     amg = ['brightness', 'contrast', 'saturation', 'RandomAdjustSharpness', 'RandomCrop', 'RandomHorizontalFlip']
    #     acept = []
    #     for i in range(len(amg)):
    #         print(f'agment List {i+1}: {amg[i]}')
    #         acept = acept.append(input(f'input 1 else 0'))
    #
    #     img = Image.open(f'{path}.png')
    #     tf = transforms.ToTensor()
    #     img = tf(img)
    #
    #     amgent = []
    #     if acept[0] ==1:
    #         amgent.append(transforms.ColorJitter((5,5), 0, 0, 0))
    #     elif acept[1]==1:
    #         amgent.append(transforms.ColorJitter(0, (5,5), 0, 0))
    #     elif acept[2]==1:
    #         amgent.append(transforms.ColorJitter(0, 0, (5,5), 0))
    #     elif acept[3]==1:
    #         amgent.append(transforms.RandomAdjustSharpness(1))
    #     elif acept[4]==1:
    #         amgent.append(transforms.RandomCrop((512,1024)))
    #     elif acept[5]==1:
    #         amgent.append(transforms.RandomHorizontalFlip(1))
    #     img_list = []
    #     print(f'{amgent}')
    #
    # visualize_agm()
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
