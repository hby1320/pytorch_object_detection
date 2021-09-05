import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from utill.utills import coords_origin_fcos
from typing import List



class FCOSHead(nn.Module):
    def __init__(self, scroe_threshold:float,
                 nms_threshold:float,
                 max_detection_box:int,
                 strides:List[int]):
        super(FCOSHead, self).__init__()
        self.scroe = scroe_threshold
        self.nms_threshold = nms_threshold
        self.max_box = max_detection_box
        self.strides = strides

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        cls_logits, coords = self._reshape_cat_out(x[0], self.strides)
        cen_logits, _ = self._reshape_cat_out(x[1], self.strides)
        reg_preds, _ = self._reshape_cat_out(x[2], self.strides)

        cls_preds = F.sigmoid(cls_logits)  # 0~1 Nomalize
        cen_preds = F.sigmoid(cen_logits)  # 0~1 Nomalize

        coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_score, cls_classes = torch.max(cls_preds, dim = -1)   #[batch, H*W]
        cls_score = torch.sqrt(cls_score * (cen_preds.squeeze(dim = -1)))
        cls_classes = cls_classes + 1

        boxes = coords_origin_fcos(coords, reg_preds)

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
        assert boxes_topk[-1] == 4
        return self.post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def post_process(self, preds_topk:List[torch.Tensor]):
        cls_scores_post = []
        cls_classes_port = []
        boxes_post = []
        cls_score_topk, cls_class_topk, box_topk = preds_topk
        for batch in range(cls_score_topk.shape[0]):
            mask = cls_score_topk >= self.scroe
            cls_scores_b = cls_score_topk[batch][mask]
            cls_classes_b = cls_class_topk[batch][mask]
            boxes_b = box_topk[batch][mask]
            nms_ind = torchvision.ops.batched_nms(boxes_b, cls_scores_b, cls_classes_b, self.nms_threshold)
            cls_scores_post.append([cls_scores_b[nms_ind]])
            cls_classes_port.append([cls_classes_b[nms_ind]])
            boxes_post.append(boxes_b[nms_ind])
        scores, classes, boxes = torch.stack(cls_scores_post, dim=0), torch.stack(cls_classes_port, dim=0), torch.\
            stack(boxes_post, dim=0)


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




