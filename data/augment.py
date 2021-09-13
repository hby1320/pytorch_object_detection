import torch
from random import random, uniform, randint
import numpy as np
from PIL import Image
import torchvision.transforms as trasforms


class Transforms(object):
    def __init__(self):
        pass

    def __call__(self, img, boxes):
        if random() < 0.3:
            img = trasforms.ColorJitter(brightness= 0.1, contrast= 0.1, saturation= 0.1, hue= 0.1)(img)
        if random() < 0.5:
            img, boxes = self.random_rotation(img, boxes)
        if random() < 0.5:
            img, boxes = random_crop_resize(img, boxes)
        return img, boxes

    # def color_jitter(self, img, boxes, brightness= 0.1, contrast= 0.1, saturation=0.1, hue=0.1):
    #     img = tra

    def random_rotation(self, img, boxes, degree:int = 10):
        degrees = uniform(-degree, degree)
        w, h = img.size
        rx0, ry0 = w / 2.0, h /2.0
        img = img.rotate(degrees)
        a = -degrees / 180.0 * np.pi
        boxes = torch.from_numpy(boxes)
        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, 0] = boxes[:, 1]
        new_boxes[:, 1] = boxes[:, 0]
        new_boxes[:, 2] = boxes[:, 3]
        new_boxes[:, 3] = boxes[:, 2]
        for i  in range(boxes.shape[0]):
            ymin, xmin, ymax, xmax = new_boxes[i, :]
            z = torch.FloatTensor([[ymin, xmin], [ymax, xmin], [ymin, xmax], [ymax, xmax]])
            tp = torch.zeros_like(z)
            tp[:, 1] = (z[:, 1] - rx0) * np.cos(a) - (z[:, 0] - ry0) * np.sin(a) + rx0
            tp[:, 0] = (z[:, 1] - rx0) * np.sin(a) - (z[:, 0] - ry0) * np.cos(a) + rx0
            ymax, xmax = torch.max(tp, dim=0)[0]
            ymin, xmin = torch.min(tp, dim=0)[0]
            new_boxes[i] = torch.stack([ymin, xmin, ymax, xmax])
        new_boxes[:, 1::2].clamp_(min=0, max= w - 1)
        new_boxes[:, 0::2].clamp_(min=0, max= h - 1)
        boxes[:, 0] = new_boxes[:, 1]
        boxes[:, 1] = new_boxes[:, 0]
        boxes[:, 3] = new_boxes[:, 3]
        boxes[:, 2] = new_boxes[:, 2]
        boxes = boxes.numpy()
        return img, boxes


def _box_inter(box1, box2):
    tl = torch.max(box1[:,None,:2], box2[:,:2])  # [n,m,2]
    br = torch.min(box1[:,None,2:], box2[:,2:])  # [n,m,2]
    hw = (br-tl).clamp(min=0)  # [n,m,2]
    inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
    return inter



def random_crop_resize(img, boxes, crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.7, attempt_max=10):
    success = False
    boxes = torch.from_numpy(boxes)
    for attempt in range(attempt_max):
        # choose crop size
        area = img.size[0] * img.size[1]
        target_area = uniform(crop_scale_min, 1.0) * area
        aspect_ratio_ = uniform(aspect_ratio[0], aspect_ratio[1])
        w = int(round(np.sqrt(target_area * aspect_ratio_)))
        h = int(round(np.sqrt(target_area / aspect_ratio_)))
        if random() < 0.5:
            w, h = h, w
        # if size is right then random crop
        if w <= img.size[0] and h <= img.size[1]:
            x = randint(0, img.size[0] - w)
            y = randint(0, img.size[1] - h)
            # check
            crop_box = torch.FloatTensor([[x, y, x + w, y + h]])
            inter = _box_inter(crop_box, boxes) # [1,N] N can be zero
            box_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]) # [N]
            mask = inter>0.0001 # [1,N] N can be zero
            inter = inter[mask] # [1,S] S can be zero
            box_area = box_area[mask.view(-1)] # [S]
            box_remain = inter.view(-1) / box_area # [S]
            if box_remain.shape[0] != 0:
                if bool(torch.min(box_remain > remain_min)):
                    success = True
                    break
            else:
                success = True
                break
    if success:
        img = img.crop((x, y, x+w, y+h))
        boxes -= torch.Tensor([x,y,x,y])
        boxes[:,1::2].clamp_(min=0, max=h-1)
        boxes[:,0::2].clamp_(min=0, max=w-1)
        # ow, oh = (size, size)
        # sw = float(ow) / img.size[0]
        # sh = float(oh) / img.size[1]
        # img = img.resize((ow,oh), Image.BILINEAR)
        # boxes *= torch.FloatTensor([sw,sh,sw,sh])
    boxes = boxes.numpy()
    return img, boxes
