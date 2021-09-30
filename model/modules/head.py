import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from utill.utills import coords_origin_fcos
from typing import List


def reshape_cat_out(inputs:torch.Tensor, strides: List[int]) -> torch.Tensor:
    '''
    Args
    inputs: list contains five [batch_size,c,_h,_w]
    Returns
    out [batch_size,sum(_h*_w),c]
    coords [sum(_h*_w),2]
    '''

    batch_size = inputs[0].shape[0]
    c = inputs[0].shape[1]

    out = []
    coords = []
    for pred, stride in zip(inputs, strides):
        pred = pred.permute(0, 2, 3, 1)
        coord = coords_origin_fcos(pred, stride).to(device=pred.device)
        pred = torch.reshape(pred, [batch_size, -1, c]) # n h*w c
        out.append(pred)
        coords.append(coord)
    return torch.cat(out, dim=1), torch.cat(coords, dim=0)


def _coords2boxes(coords, offsets):
    '''
    Args
    coords [sum(_h*_w),2]
    offsets [batch_size,sum(_h*_w),4] ltrb
    '''
    x1y1 = coords[None, :, :] - offsets[..., :2]
    x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
    boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
    return boxes


class FCOSHead(nn.Module):
    def __init__(self, score_threshold: float,
                 nms_threshold: float,
                 max_detection_box: int,
                 strides: List[int]):
        super(FCOSHead, self).__init__()
        self.score = score_threshold
        self.nms_threshold = nms_threshold
        self.max_box = max_detection_box
        self.strides = strides

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_logits, coords = reshape_cat_out(x[0], self.strides)
        cen_logits, _ = reshape_cat_out(x[1], self.strides)
        reg_preds, _ = reshape_cat_out(x[2], self.strides)

        cls_preds = torch.sigmoid(cls_logits)  # 0~1 Nomalize
        cen_preds = torch.sigmoid(cen_logits)  # 0~1 Nomalize

        coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_score, cls_classes = torch.max(cls_preds, dim = -1)   #[batch, H*W]
        cls_score = torch.sqrt(cls_score * (cen_preds.squeeze(dim = -1)))
        cls_classes = cls_classes + 1

        boxes = _coords2boxes(coords, reg_preds)

        # select top-k
        max_num = min(self.max_box, cls_score.shape[-1])
        topk_ind = torch.topk(cls_score, max_num, dim = 1, largest = True, sorted=True)[1]  # batch,max_num
        cls_scores = []
        cls_class = []
        boxs = []
        for batch in range(cls_score.shape[0]):
            cls_scores.append(cls_score[batch][topk_ind[batch]])
            cls_class.append(cls_classes[batch][topk_ind[batch]])
            boxs.append(boxes[batch][topk_ind[batch]])
        cls_scores_topk = torch.stack(cls_scores, dim=0)
        cls_classes_topk = torch.stack(cls_class, dim=0)
        boxes_topk = torch.stack(boxs, dim=0)
        assert boxes_topk.shape[-1] == 4
        return self.post_process([cls_scores_topk, cls_classes_topk, boxes_topk])


    def post_process(self, preds_topk:List[torch.Tensor]):
        cls_scores_post = []
        cls_classes_port = []
        boxes_post = []
        cls_score_topk, cls_class_topk, box_topk = preds_topk
        for batch in range(cls_score_topk.shape[0]):
            mask = cls_score_topk[batch] >= self.score
            cls_scores_b = cls_score_topk[batch][mask]
            cls_classes_b = cls_class_topk[batch][mask]
            boxes_b = box_topk[batch][mask]
            nms_ind = torchvision.ops.batched_nms(boxes_b, cls_scores_b, cls_classes_b, self.nms_threshold)
            # nms_ind = self.batched_nms(boxes_b, cls_scores_b, cls_classes_b, self.nms_threshold)
            cls_scores_post.append(cls_scores_b[nms_ind])
            cls_classes_port.append(cls_classes_b[nms_ind])
            boxes_post.append(boxes_b[nms_ind])
            # print(f'{cls_scores_post=}\n{cls_classes_port}\n{boxes_post}')
            # print(cls_scores_post)
        scores, classes, boxes = torch.stack(cls_scores_post, dim=0), torch.stack(cls_classes_port, dim=0), torch.stack(boxes_post, dim=0)
        return scores, classes, boxes

    # def batched_nms(self, boxes, scores, idxs, iou_threshold):
    #
    #     if boxes.numel() == 0:
    #         return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    #     # strategy: in order to perform NMS independently per class.
    #     # we add an offset to all the boxes. The offset is dependent
    #     # only on the class idx, and is large enough so that boxes
    #     # from different classes do not overlap
    #     max_coordinate = boxes.max()
    #     offsets = idxs.to(boxes) * (max_coordinate + 1)
    #     boxes_for_nms = boxes + offsets[:, None]
    #     keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
    #     return keep
    #
    # @staticmethod
    # def box_nms(boxes, scores, thr):
    #     '''
    #     boxes: [?,4]
    #     scores: [?]
    #     '''
    #     if boxes.shape[0] == 0:
    #         return torch.zeros(0, device=boxes.device).long()
    #     assert boxes.shape[-1] == 4
    #     x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    #     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #     order = scores.sort(0, descending=True)[1]
    #     keep = []
    #     while order.numel() > 0:
    #         if order.numel() == 1:
    #             i = order.item()
    #             keep.append(i)
    #             break
    #         else:
    #             i = order[0].item()
    #             keep.append(i)
    #
    #         xmin = x1[order[1:]].clamp(min=float(x1[i]))
    #         ymin = y1[order[1:]].clamp(min=float(y1[i]))
    #         xmax = x2[order[1:]].clamp(max=float(x2[i]))
    #         ymax = y2[order[1:]].clamp(max=float(y2[i]))
    #         inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
    #         iou = inter / (areas[i] + areas[order[1:]] - inter)
    #         idx = (iou <= thr).nonzero().squeeze()
    #         if idx.numel() == 0:
    #             break
    #         order = order[idx + 1]
    #     return torch.LongTensor(keep)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp_(min = 0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max = w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max = h - 1)
        return batch_boxes

    # def batched_nms(self, boxes, scores, idxs, iou_threshold):
    #
    #     if boxes.numel() == 0: ## 원소의 개수확인
    #         return torch.empty((0,), dtype = torch.int64, device = boxes.device)
    #
    #     max_coordinate = boxes.max()
    #     offsets = idxs.to(boxes) * (max_coordinate +1)
    #     boxes_for_nms = boxes + offsets[:, None]
    #     keep = torchvision.ops.nms()
    #     torchvision.ops.batched_nms()




