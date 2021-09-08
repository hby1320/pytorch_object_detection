import torch
from random import random, uniform
import numpy as np
from PIL import Image
import torchvision.transforms as trasforms


class Transforms(object):
    def __init__(self):
        pass

    def __call__(self, img, boxes):
        if random < 0.3:
            img = trasforms.ColorJitter(brightness= 0.1, contrast= 0.1, saturation= 0.1, hue= 0.1)(img)
        if random < 0.5:
            img, boxes = self.random_rotation(img, boxes)
        if random < 0.5:
            img, boxes = self.random_crop_resize(img, boxes)
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
            ymin, xmin, ymax, xmax = float(new_boxes[i, :])
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









