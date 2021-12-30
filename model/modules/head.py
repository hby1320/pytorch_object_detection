import torch
import torch.nn as nn
import torchvision.ops
from utill.utills import coords_origin_fcos
from typing import List


def reshape_cat_out(inputs: torch.Tensor, strides: List[int]) -> torch.Tensor:
    """
    Args
    inputs: list contains five [batch_size,c,_h,_w]
    Returns
    out [batch_size,sum(_h*_w),c]
    coords [sum(_h*_w),2]
    """
    out = []
    coords = []
    batch_size = inputs[0].shape[0]
    c = inputs[0].shape[1]
    for pred, stride in zip(inputs, strides):
        pred = pred.permute(0, 2, 3, 1)  # B, H, W, C
        coord = coords_origin_fcos(pred, stride).to(device=pred.device)  # center point
        pred = torch.reshape(pred, [batch_size, -1, c])  # B, H*W, C
        out.append(pred)
        coords.append(coord)
    return torch.cat(out, dim=1), torch.cat(coords, dim=0)


def _coords2boxes(coords, offsets):
    """
    Args
    coords [sum(_h*_w),2]
    offsets [batch_size,sum(_h*_w),4] l,t, r,b
    """
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
        cls_logits, coords = reshape_cat_out(x[0], self.strides)  # cls,
        cen_logits, _ = reshape_cat_out(x[1], self.strides)  # cnt
        reg_preds, _ = reshape_cat_out(x[2], self.strides)  # reg

        cls_preds = torch.sigmoid(cls_logits)  # 0~1 normalized
        cen_preds = torch.sigmoid(cen_logits)  # 0~1 normalized

        coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_score, cls_classes = torch.max(cls_preds, dim=-1)   # [batch, H*W]
        cls_score = torch.sqrt(cls_score * (cen_preds.squeeze(dim=-1)))
        cls_classes = cls_classes + 1  # background del

        boxes = _coords2boxes(coords, reg_preds)

        # select top-k
        max_num = min(self.max_box, cls_score.shape[-1])
        top_k_ind = torch.topk(cls_score, max_num, dim=1, largest=True, sorted=True)[1]  # batch,max_num
        cls_scores = []
        cls_class = []
        boxs = []
        for batch in range(cls_score.shape[0]):
            cls_scores.append(cls_score[batch][top_k_ind[batch]])
            cls_class.append(cls_classes[batch][top_k_ind[batch]])
            boxs.append(boxes[batch][top_k_ind[batch]])
        cls_scores_top_k = torch.stack(cls_scores, dim=0)
        cls_classes_top_k = torch.stack(cls_class, dim=0)
        boxes_top_k = torch.stack(boxs, dim=0)
        assert boxes_top_k.shape[-1] == 4
        return self.post_process([cls_scores_top_k, cls_classes_top_k, boxes_top_k])

    def post_process(self, preds_top_k: List[torch.Tensor]):
        cls_scores_post = []
        cls_classes_port = []
        boxes_post = []
        cls_score_top_k, cls_class_top_k, box_top_k = preds_top_k
        for batch in range(cls_score_top_k.shape[0]):
            mask = cls_score_top_k[batch] >= self.score
            cls_scores_b = cls_score_top_k[batch][mask]
            cls_classes_b = cls_class_top_k[batch][mask]
            boxes_b = box_top_k[batch][mask]
            nms_ind = torchvision.ops.batched_nms(boxes_b, cls_scores_b, cls_classes_b, self.nms_threshold)
            # nms_ind = self.batched_nms(boxes_b, cls_scores_b, cls_classes_b, self.nms_threshold)
            cls_scores_post.append(cls_scores_b[nms_ind])
            cls_classes_port.append(cls_classes_b[nms_ind])
            boxes_post.append(boxes_b[nms_ind])
        scores, classes, boxes = torch.stack(cls_scores_post, dim=0), \
                                 torch.stack(cls_classes_port, dim=0), \
                                 torch.stack(boxes_post, dim=0)
        return scores, classes, boxes

    # def batched_nms(self, boxes, scores, idxs, iou_threshold):
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

    @staticmethod
    def forward(batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp_(min=0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
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


class DefaultBoxGenerator(nn.Module):
    def __init__(self, aspect_ratios, max_ratio, min_ratio, steps, clip, scales):
        super(DefaultBoxGenerator, self).__init__()
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip
        number_of_output = len(aspect_ratios)

        if scales is None:
            if number_of_output > 1:
                range_ratio = max_ratio - min_ratio
                self.scales = [min_ratio + range_ratio * k / (number_of_output - 1.0) for k in range(number_of_output)]
                self.scales.append(1.0)
            else:
                self.scales = scales

            self.wh_pairs = self.generates_wh_paris(number_of_output)

    def generates_wh_paris(self, number_of_output):
        wh_pairs = []
        for k in range(number_of_output):
            s_k = self.scales[k]
            s_prime_k = torch.sqrt(self.scales[k] * self.scales[k+1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            for ar in self.aspect_ratios[k]:
                sq_ar = torch.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h], [h, w]])

            wh_pairs.append(torch.as_tensor(wh_pairs))
        return wh_pairs


class FCOSGenTargets(nn.Module):
    def __init__(self, strides: List[int], limit_range: List[List[int]]):
        super(FCOSGenTargets, self).__init__()
        self.stride = strides
        self.lim_range = limit_range
        assert len(strides) == len(limit_range)  # s와 범위 same

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_logit, center_logit, reg_logit = x[0]
        gt_box = x[1]
        labels = x[2]
        cls_target = []
        center_target = []
        reg_target = []
        assert len(self.stride) == len(cls_logit)
        for lv in range(len(cls_logit)):
            lv_out = [cls_logit[lv], center_logit[lv], reg_logit[lv]]  # output shape 8 16 32 64 128
            level_targets = self.generate_target(lv_out, gt_box, labels, self.stride[lv], self.lim_range[lv])
            cls_target.append(level_targets[0])
            center_target.append(level_targets[1])
            reg_target.append(level_targets[2])
        return torch.cat(cls_target, dim=1), torch.cat(center_target, dim=1), torch.cat(reg_target, dim=1)

    @staticmethod
    def generate_target(lv_out: torch.Tensor,
                        gt_box: torch.Tensor,
                        labels: torch.Tensor,
                        stride: List[int],
                        lim_range: List[List[int]],
                        sample_radio_ratio: float = 1.5) -> torch.Tensor:
        """
        :param lv_out: class, cnt, reg
        :param gt_box: [N, M, 4]
        :param labels: [N, M, C]
        :param stride: [8, 16, 32, 64, 128]
        :param lim_range:
        :param sample_radio_ratio: default 1.5
        :return: torch.Tensor
        """

        cls_logit, center_logit, reg_logit = lv_out  # [8, 16, 32, 64, 128]
        batch = cls_logit.shape[0]  # [N,C,H,W]
        class_num = cls_logit.shape[1]
        cls_logit = cls_logit.permute(0, 2, 3, 1)  # b,n,h,w -> b,h,w,c
        coords = coords_origin_fcos(feature=cls_logit, strides=stride).to(device=gt_box.device)  # [H*W , 2]
        cls_logit = cls_logit.reshape((batch, -1, class_num))  # [N, H*W, C]

        hw = cls_logit.shape[1]  # [N, H*W, C]
        x = coords[:, 0]  # [H*W , 0] X
        y = coords[:, 1]  # [H*W , 0] Y
        left_offset = x[None, :, None] - gt_box[..., 0][:, None, :]  # x:[1, H*W, 1] - gt_box: [N,1,4]
        top_offset = y[None, :, None] - gt_box[..., 1][:, None, :]
        right_offset = gt_box[..., 2][:, None, :] - x[None, :, None]
        bottom_offset = gt_box[..., 3][:, None, :] - y[None, :, None]
        offset = torch.stack([left_offset, top_offset, right_offset, bottom_offset], dim=-1)

        #  torch.Size([20, 4096, 4, 4])
        area = (offset[..., 0]+offset[..., 2]) * (offset[..., 1]+offset[..., 3])  # [Class, H*W, M]
        offset_min = torch.min(offset, dim=-1)[0]  # [Class, H*W, M]min[0] -> Value
        offset_max = torch.max(offset, dim=-1)[0]

        mask_gt = offset_min > 0
        mask_lv = (offset_max > lim_range[0]) & (offset_max <= lim_range[1])

        ratio = stride * sample_radio_ratio
        gt_center_x = (gt_box[..., 0] + gt_box[..., 2]) / 2  # 중심 생성
        gt_center_y = (gt_box[..., 1] + gt_box[..., 3]) / 2
        gt_left_offset = x[None, :, None] - gt_center_x[:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        gt_top_offset = y[None, :, None] - gt_center_y[:, None, :]
        gt_right_offset = gt_center_x[:, None, :] - x[None, :, None]
        gt_bottom_offset = gt_center_y[:, None, :] - y[None, :, None]
        gt_offset = torch.stack([gt_left_offset, gt_top_offset, gt_right_offset, gt_bottom_offset], dim=-1)
        gt_off_max = torch.max(gt_offset, dim=-1)[0]
        mask_center = gt_off_max < ratio

        mask_pos = mask_gt & mask_lv & mask_center  #

        area[~mask_pos] = 99999999
        area_min_index = torch.min(area, dim=-1)[1]  # area min val index
        reg_target = offset[torch.zeros_like(area, dtype=torch.bool).scatter_(-1, area_min_index.unsqueeze(dim=-1), 1)]
        reg_target = torch.reshape(reg_target, (batch, -1, 4))

        labels = torch.broadcast_tensors(labels[:, None, :], area.long())[0]  # int 형 탠서 사용
        cls_target = labels[torch.zeros_like(area, dtype=torch.bool).scatter_(-1, area_min_index.unsqueeze(dim=-1), 1)]
        cls_target = torch.reshape(cls_target, (batch, -1, 1))

        left_right_min = torch.min(reg_target[..., 0], reg_target[..., 2])  # [batch_size,h*w]
        left_right_max = torch.max(reg_target[..., 0], reg_target[..., 2])
        top_bottom_min = torch.min(reg_target[..., 1], reg_target[..., 3])
        top_bottom_max = torch.max(reg_target[..., 1], reg_target[..., 3])
        center_target = (
                (left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim=-1)

        assert reg_target.shape == (batch, hw, 4)
        assert cls_target.shape == (batch, hw, 1)
        assert center_target.shape == (batch, hw, 1)

        mask_pos_2 = mask_pos.long().sum(dim=-1)

        mask_pos_2 = mask_pos_2 >= 1
        assert mask_pos_2.shape == (batch, hw)
        cls_target[~mask_pos_2] = 0
        center_target[~mask_pos_2] = -1
        reg_target[~mask_pos_2] = -1

        return cls_target, center_target, reg_target
