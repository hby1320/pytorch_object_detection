import torch
import torch.nn as nn
import numpy as np
from typing import List
from torchinfo import summary
from thop import profile


def model_info(model: nn.Module, batch: int, ch: int, width: int, hight: int, device: torch.device, depth=4):
    col_names = ["input_size", "output_size", "num_params", "kernel_size", "mult_adds"]
    img = torch.rand(batch, ch, width, hight).to(device)
    summary(model, img.size(), None, None, col_names=col_names, depth = depth, verbose= 1)
    flop, para = profile(model, inputs=(img,))
    print(f'flop{(flop/1e9):.2f}G  para{(para/1e9):.2f}G')


def generate_anchor(base_size=16, ratios=None, scales=None):
    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchor = len(ratios) * len(scales)
    anchor = np.zeros((num_anchor, 4))
    anchor[:, :2] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchor[:, 2] * anchor[:, 3]

    anchor[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchor[:, 3] = anchor[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchor[:, 0::2] -= np.tile(anchor[:,2] * 0.5, (2,1)).T
    anchor[:, 1::2] -= np.tile(anchor[:, 2] * 0.5, (2, 1)).T

    return anchor


def shift_xy(shape, stride, anchor):
    shift_x = (np.arange(0, shape[1] + 0.5) * stride)
    shift_y = (np.arange(0, shape[0] + 0.5) * stride)

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    ))

    A = anchor.shape[0]
    K = shifts.shape[0]
    all_anchor = (anchor.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchor = all_anchor.reshape((K*A, 4))

    return all_anchor


def coords_origin_fcos(feature: torch.Tensor, strides: List[int]) -> torch.Tensor:  # 원본 FCOS의 Location과 동일
    """
    :param feature:
    :param strides:
    :return:
    """
    h, w = feature.shape[1:3]  # [N, H, W, C] -> H,W

    shifts_x = torch.arange(0, w * strides, strides, dtype=torch.float32)  # stride 만큼 간격
    shifts_y = torch.arange(0, h * strides, strides, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])  # Number of pixel  -> [H*W]
    shift_y = torch.reshape(shift_y, [-1])  # Number of pixel  -> [H*W]
    coords = torch.stack([shift_x, shift_y], dim=-1) + strides // 2  # Number of pixel  -> [H*W]
    return coords


def voc_collect(samples: torch.Tensor):
    images = [sample['img'] for sample in samples]
    targets = [sample['targets'] for sample in samples]
    lables = [sample['lables'] for sample in samples]
    padded_imgs = torch.nn.utils.rnn.pad_sequence(images, batch_first=True)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    padded_tlables = torch.nn.utils.rnn.pad_sequence(lables, batch_first=True)

    return padded_imgs, padded_targets, padded_tlables


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, max_iter: int, power=0.9, min_lr=1e-6, last_epoch=-1):
        assert max_iter != 0
        self.max_iter = max_iter
        self.power = power
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iter) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


class DataEncoder:
    def __init__(self):
        self.anchor_areas = [32 * 32., 64 * 64., 128 * 128., 256 * 256., 512 * 512.]  # 피쳐맵 크기 p3 -> p7
        self.aspect_ratios = [1 / 2., 1 / 1., 2 / 1.]  # 앵커 박스 종횡비, w/h
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]  # 앵커 박스 scale
        self.anchor_wh = self._get_anchor_wh()  # 5개의 피쳐맵 각각에 해당하는 9개의 앵커 박스 생성

    def _get_anchor_wh(self):
        # 각 피쳐맵에서 사용할 앵커 박스 높이와 넓이를 계산합니다.
        anchor_wh = []
        for s in self.anchor_areas:  # 각 피쳐맵 크기 추출
            for ar in self.aspect_ratios:  # ar = w/h
                h = np.sqrt(s / ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)  # [#fms, #anchors_pre_cell, 2], [5, 9, 2]

    def _get_anchor_boxes(self, input_size):
        # 피쳐맵의 모든 cell에 앵커 박스 할당
        num_fms = len(self.anchor_areas)  # 5
        fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in range(num_fms)]  # 각 피쳐맵 stride 만큼 입력 크기 축소

        boxes = []
        for i in range(num_fms):  # p3 ~ p7
            fm_size = fm_sizes[i]  # i 번째 피쳐맵 크기 추출
            grid_size = input_size / fm_size  # 입력 크기를 피쳐맵 크기로 나누어 grid size 생성
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = self._meshgrid(fm_w, fm_h) + 0.5  # [fm_h * fm_w, 2] 피쳐맵 cell index 생성
            xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 9, 2)  # anchor 박스 좌표
            wh = self.anchor_wh[i].view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2)  # anchor 박스 높이와 너비
            box = torch.cat([xy, wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0)

    # 피쳐맵의 각 셀에 anchor 박스 생성하고, positive와 negative 할당
    def encode(self, boxes, labels, input_size):
        input_size = torch.Tensor([input_size, input_size]) if isinstance(input_size, int) else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)  # 앵커 박스 생성
        boxes = self._change_box_order(boxes, 'xyxy2xywh')  # xyxy -> cxcywh

        ious = self._box_iou(anchor_boxes, boxes, order = 'xywh')  # ground-truth와 anchor의 iou 계산
        max_ious, max_ids = ious.max(1)  # 가장 높은 iou를 지닌 앵커 추출
        boxes = boxes[max_ids]

        # 앵커 박스와의 offset 계산
        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)

        # class 할당
        cls_targets = 1 + labels[max_ids]
        cls_targets[max_ious < 0.5] = 0  # iou < 0.5 anchor는 negative
        ignore = (max_ious > 0.4) & (max_ious < 0.5)  # [0.4,0.5] 는 무시
        cls_targets[ignore] = -1
        return loc_targets, cls_targets

    # encode된 값을 원래대로 복구 및 nms 진행
    def decode(self, loc_preds, cls_preds, input_size):
        cls_thresh = 0.5
        nms_thresh = 0.5

        input_size = torch.Tensor([input_size, input_size]) if isinstance(input_size, int) else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)  # 앵커 박스 생성

        loc_xy = loc_preds[:, :2]  # 결과값 offset 추출
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]  # offset + anchor
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        score, labels = cls_preds.sigmoid().max(1)
        ids = score > cls_thresh
        ids = ids.nonzero().squeeze()
        keep = self._box_nms(boxes[ids], score[ids], threshold = nms_thresh)  # nms
        return boxes[ids][keep], labels[ids][keep]

    # cell index 생성 함수
    def _meshgrid(self, x, y, row_major=True):
        a = torch.arange(0, x)
        b = torch.arange(0, y)
        xx = a.repeat(y).view(-1, 1)
        yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
        return torch.cat([xx, yy], 1) if row_major else torch.cat([yy, xx], 1)

    # x1,y1,x2,y2 <-> cx,cy,w,h
    def _change_box_order(self, boxes, order):
        assert order in ['xyxy2xywh', 'xywh2xyxy']
        boxes = np.array(boxes)
        a = boxes[:, :2]
        b = boxes[:, 2:]
        a, b = torch.Tensor(a), torch.Tensor(b)
        if order == 'xyxy2xywh':
            return torch.cat([(a + b) / 2, b - a + 1], 1)  # xywh
        return torch.cat([a - b / 2, a + b / 2], 1)  # xyxy

    # 두 박스의 iou 계산
    def _box_iou(self, box1, box2, order='xyxy'):
        if order == 'xywh':
            box1 = self._change_box_order(box1, 'xywh2xyxy')
            box2 = self._change_box_order(box2, 'xywh2xyxy')

        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])

        wh = (rb - lt + 1).clamp(min = 0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
        area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)
        iou = inter / (area1[:, None] + area2 - inter)
        return iou

    # nms
    def _box_nms(self, bboxes, scores, threshold=0.5, mode='union'):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        _, order = scores.sort(0, descending = True)  # confidence 순 정렬
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.data)
                break
            i = order[0]  # confidence 가장 높은 anchor 추출
            keep.append(i)  # 최종 detection에 저장

            xx1 = x1[order[1:]].clamp(min = x1[i])
            yy1 = y1[order[1:]].clamp(min = y1[i])
            xx2 = x2[order[1:]].clamp(max = x2[i])
            yy2 = y2[order[1:]].clamp(max = y2[i])

            w = (xx2 - xx1 + 1).clamp(min = 0)
            h = (yy2 - yy1 + 1).clamp(min = 0)
            inter = w * h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max = areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        return torch.LongTensor(keep)
