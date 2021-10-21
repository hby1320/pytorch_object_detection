import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, smooth_l1_loss, l1_loss, huber_loss
# from torch.nn import L1Loss


def compute_cls_loss(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.tensor:
    batch_size = target.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1] # 20 / 80
    mask = mask.unsqueeze(dim = -1)   #  torch.Size([2, 4724, 1])
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, class_num])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)  # [batch_size,sum(_h*_w),class_num]
    assert preds.shape[:2] == target.shape[:2]
    cls_loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
        target_pos = target[batch_index]  # [sum(_h*_w),1]
        target_pos = (torch.arange(1, class_num + 1, device = target_pos.device)[None, :] == target_pos).float()
        # sparse--> one-hot
        cls_loss.append(focal_loss_from_logits(pred_pos, target_pos).view(1))
    return torch.cat(cls_loss, dim=0)/num_pos  # [batch_size,]


def compute_cnt_loss(preds, target, mask):
    batch_size = target.shape[0]
    c = target.shape[-1]
    preds_reshape = []
    mask = mask.unsqueeze(dim=-1)
    # mask= target>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim = [1, 2]).clamp_(min = 1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim = 1) ## 차원 증가
    assert preds.shape == target.shape  # [batch_size,sum(_h*_w),1]
    cnt_loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,]
        target_pos = target[batch_index][mask[batch_index]]  # [num_pos_b,]
        assert len(pred_pos.shape) == 1
        cnt_loss.append(binary_cross_entropy_with_logits(input=pred_pos,
                                                     target=target_pos,
                                                     reduction='sum').view(1))
    return torch.cat(cnt_loss, dim = 0)/num_pos  # [batch_size,]


def compute_dcnt_loss(preds, reg_target, cnt_target, mask):
    batch_size = cnt_target.shape[0]
    # c = cnt_target.shape[-1]
    c=5
    preds_reshape = []
    cnt_mask = mask.unsqueeze(dim=-1)
    # mask= target>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(cnt_mask, dim = [1, 2]).clamp_(min = 1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim = 1) ## 차원 증가
    #assert preds.shape == cnt_target.shape  # [batch_size,sum(_h*_w),1]
    cnt_loss = []
    distance_loss = []
    total_loss = []
    for batch_index in range(batch_size):
        pred_cnt_pos = preds[batch_index, :, 0][cnt_mask[batch_index][..., 0]]# [num_pos_b,1]
        cnt_target_pos = cnt_target[batch_index][cnt_mask[batch_index]]  # [num_pos_b,1]
        pred_reg_pos = preds[batch_index, :, 1:5][mask[batch_index]]  # [num_pos_b,4]
        reg_target_pos = reg_target[batch_index][mask[batch_index]]  # [num_pos_b,4]
        assert len(pred_cnt_pos.shape) == 1
        assert len(pred_reg_pos.shape) == 2

        # lt = torch.min(pred_reg_pos[..., :2], dim=1)
        # rb = torch.min(pred_reg_pos[..., 2:], dim=1)
        # distance = torch.sqrt(torch.pow(rb[0],2)+torch.pow(lt[0],2)).clamp(min=0)
        #
        # gt_lt = torch.min(reg_target_pos[..., :2], dim=1)
        # gt_rb = torch.min(reg_target_pos[..., 2:], dim=1)
        # gt_distance = torch.sqrt(torch.pow(gt_lt[0],2) + torch.pow(gt_rb[0],2)).clamp(min=0)
        w = pred_reg_pos[..., 0] + pred_reg_pos[..., 2]
        h = pred_reg_pos[..., 1] + pred_reg_pos[..., 3]
        distance = torch.sqrt(torch.pow(w, 2) + torch.pow(h, 2)).clamp(min=0)
        distance = torch.sigmoid(distance)
        gtw = reg_target_pos[..., 0] + reg_target_pos[..., 2]
        gth = reg_target_pos[..., 1] + reg_target_pos[..., 3]
        gt_distance = torch.sqrt(torch.pow(gtw, 2) + torch.pow(gth, 2)).clamp(min=0)
        gt_distance = torch.sigmoid(gt_distance)
        cnt_loss.append((binary_cross_entropy_with_logits(input=pred_cnt_pos,
                                                         target=cnt_target_pos,
                                                         reduction='sum')+
                         (0.5 * l1_loss(input=distance, target=gt_distance, reduction='sum'))).view(1))
        # distance_loss.append(0.2*smooth_l1_loss(input=distance,
        #                                target=gt_distance,
        #                                reduction='sum').view(1))
        # total_loss.append(cnt_loss+distance_loss)
        # dcnt_loss = torch.cat([cnt_loss, distance])
        # cnt_loss.append(smooth_l1_loss(input=wh,
        # #                                          target=gt_wh,
        # #                                          reduction='sum').view(1)
        # #                 )
    return torch.cat(cnt_loss, dim=0) / num_pos  # [batch_size,]


def compute_reg_loss(preds, target, mask, mode='iou'):
    batch_size = target.shape[0]
    c = target.shape[-1]
    preds_reshape = []
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos = torch.sum(mask, dim = 1).clamp_(min = 1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == target.shape  # [batch_size,sum(_h*_w),4]
    reg_loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,4]
        target_pos = target[batch_index][mask[batch_index]]  # [num_pos_b,4]
        assert len(pred_pos.shape) == 2
        if mode == 'iou':
            reg_loss.append(iou_loss(pred_pos, target_pos).view(1))
        elif mode == 'giou':
            reg_loss.append(giou_loss(pred_pos, target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(reg_loss, dim = 0)/num_pos  # [batch_size,]


def iou_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    lt = torch.min(preds[:, :2], targets[:, :2])
    rb = torch.min(preds[:, 2:], targets[:, 2:])
    wh = torch.clamp((rb + lt), min=0)
    # wh = (rb + lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    iou = overlap / (area1 + area2 - overlap)
    loss_iou = -iou.clamp(min=1e-6).log()
    return loss_iou.sum()


def giou_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    :param preds:
    :param targets:
    :return:
    """
    lt_min = torch.min(preds[:, :2], targets[:, :2])
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = torch.clamp((rb_min + lt_min), min=0)
    overlap = wh_min[:, 0] * wh_min[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    union = (area1 + area2 - overlap)
    iou = overlap/union

    lt_max = torch.max(preds[:, :2], targets[:, :2])
    rb_max = torch.max(preds[:, 2:], targets[:, 2:])
    wh_max = torch.clamp((rb_max + lt_max), min = 0)
    G_area = wh_max[:, 0] * wh_max[:, 1]  # [n]

    giou = iou-(G_area - union)/G_area.clamp(1e-10)
    loss_giou = 1. - giou
    return loss_giou.sum()


def focal_loss_from_logits(preds: torch.Tensor, targets: torch.Tensor, gamma=2.0, alpha=0.25):
    """
    :param preds:
    :param targets:
    :param gamma:
    :param alpha:
    :return:
    """
    preds = preds.sigmoid()
    preds = torch.clip(preds, min=0.000005, max=0.99999999995)
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_loss = -w * torch.pow((1.0 - pt), gamma) * pt.log()
    return focal_loss.sum()


class FCOSLoss(nn.Module):
    def __init__(self, mode: str = 'giou'):
        super(FCOSLoss, self).__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:
        :return:
        """
        pred, target = x
        cls_logit, cnt_logit, reg_logit = pred
        cls_target, cnt_target, reg_target = target
        mask_pos = (cnt_target > -1).squeeze(dim=-1)  ## squeeze dim 1 제거
        cls_loss = compute_cls_loss(cls_logit, cls_target, mask_pos).mean()  #평균 이유 : 배치 당 로스
        cnt_loss = compute_cnt_loss(cnt_logit, cnt_target, mask_pos).mean()
        # cnt_loss = compute_dcnt_loss(cnt_logit, reg_target, cnt_target, mask_pos).mean()
        reg_loss = compute_reg_loss(reg_logit, reg_target, mask_pos, self.mode).mean()
        total_loss = cls_loss + cnt_loss + reg_loss
        return cls_loss, cnt_loss, reg_loss, total_loss


if __name__ == '__main__':
    loss = compute_cnt_loss([torch.ones([2, 1, 4, 4])] * 5, torch.ones([2, 80, 1]),
                            torch.ones([2, 80], dtype = torch.bool))
    print(loss)   # tensor([0.3133, 0.3133])
    # loss = compute_dcnt_loss([torch.ones([2, 4, 4, 5])] * 5, torch.ones([2, 80, 1]),
    #                         torch.ones([2, 80], dtype=torch.bool))
    # print(loss)  # tensor([0.3133, 0.3133])