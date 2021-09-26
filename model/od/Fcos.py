import numpy as np
import torch
import torch.nn as nn
from utill.utills import model_info, coords_origin_fcos
from model.backbone.resnet50 import ResNet50
# from model.backbone.resnet import resnet50
import torch.nn.functional as F
from typing import List
import numpy as np


class FCOS(nn.Module):
    """
    Total params: 38,963,484
    Trainable params: 35,920,348
    Non-trainable params: 3,043,136
    Total mult-adds (G): 149.93

    Input size (MB): 3.158.07
    Forward/backward pass size (MB): 1198.24
    Params size (MB): 155.85
    """
    def __init__(self, in_channel: List[int], num_class: int, feature: int, freeze_bn: bool = True):
        super(FCOS, self).__init__()
        self.backbone = ResNet50(3)
        # self.backbone = resnet50(pretrained=True)
        self.FPN = FeaturePyramidNetwork(in_channel, feature)
        self.head = HeadFCOS(feature, num_class, 0.01)
        self.backbone_freeze = freeze_bn

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters():
                    p.requires_grad = False  # 학습 x
        #
        if self.backbone_freeze:
            self.apply(freeze_bn)
            self.backbone.freeze_stages(1)
            print(f"success frozen BN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.FPN(x)
        cls, cnt, reg = self.head(x)
        # cls = []
        # reg = []
        # center = []
        # for i, feature in enumerate(x):
        #     cls_logit = self.classification_sub(feature)
        #     center_logit, reg_logit = self.regression_sub(feature)
        #     cls.append(cls_logit)
        #     center.append(center_logit)
        #     reg.append(self.scale_exp[i](reg_logit))
        return [cls, cnt, reg]


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channel:List[int], feature=256):
        super(FeaturePyramidNetwork, self).__init__()
        self.P5 = nn.Conv2d(in_channels = in_channel[0], out_channels = feature, kernel_size = 1)
        self.P4 = nn.Conv2d(in_channels = in_channel[1], out_channels = feature, kernel_size = 1)
        self.P3 = nn.Conv2d(in_channels = in_channel[2], out_channels = feature, kernel_size = 1)
        self.P5_Up = nn.Upsample(scale_factor = 2)
        self.P4_Up = nn.Upsample(scale_factor = 2)
        self.P5_c1 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1)
        self.P4_c1 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1)
        self.P3_c1 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1)
        self.P6_c1 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, stride = 2)
        self.P7_c1 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, stride = 2)
        self.act = nn.ReLU(True)
        self.apply(self.init_conv_kaiming)

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3, c4, c5 = x
        p5 = self.P5(c5)
        p4_c = self.P4(c4)
        p3_c = self.P3(c3)

        p4 = self.P5_Up(p5)
        p4 = torch.add(p4, p4_c)

        p3 = self.P4_Up(p4)
        p3 = torch.add(p3, p3_c)

        p3 = self.P3_c1(p3)
        p4 = self.P4_c1(p4)
        p5 = self.P5_c1(p5)
        # p6 = self.P6_c1(p5)
        # p7 = self.P7_c1(self.act(p6))
        return [p3, p4, p5]


class HeadFCOS(nn.Module):
    def __init__(self, feature:int, num_class: int, prior: float = 0.01):
        super(HeadFCOS, self).__init__()
        self.class_num = num_class
        self.prior = prior
        cls_branch = []
        reg_branch = []

        for i in range(2):
            cls_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False))
            cls_branch.append(nn.GroupNorm(32, feature))
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False))
            reg_branch.append(nn.GroupNorm(32, feature))
            reg_branch.append(nn.ReLU(True))

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(feature, self.class_num, kernel_size=3, padding=1)
        self.cnt_logits = nn.Conv2d(feature, 1, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(feature, 4, kernel_size=3, padding=1)

        self.apply(self.init_conv_RandomNormal)

        nn.init.constant_(self.cls_logits.bias, -np.log((1 - self.prior) / self.prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(3)])

    def init_conv_RandomNormal(self, module, std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        for index, P in enumerate(inputs):
            cls_conv_out = self.cls_conv(P)
            reg_conv_out = self.reg_conv(P)
            cls_logits.append(self.cls_logits(cls_conv_out))
            cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds


# class ClassificationSub(nn.Module):
#     def __init__(self, feature, num_class, prior):
#         super(ClassificationSub, self).__init__()
#         self.prior = prior
#         self.cls_c1 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
#         self.cls_c2 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
#         self.cls_c3 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
#         self.cls_c4 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
#         self.cls = nn.Conv2d(feature, num_class, kernel_size = 3, padding = 1)
#         # self.center = nn.Conv2d(feature, 1, kernel_size = 3, padding = 1)
#         self.gn = nn.GroupNorm(32, feature)
#         self.gn2 = nn.GroupNorm(32, feature)
#         self.gn3 = nn.GroupNorm(32, feature)
#         self.gn4 = nn.GroupNorm(32, feature)
#         self.relu = nn.ReLU(inplace=True)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.relu4 = nn.ReLU(inplace=True)
#         self.apply(init_conv_rand_nomal)
#         nn.init.constant_(self.cls.bias, -np.log((1-self.prior) / self.prior))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.relu(self.gn(self.cls_c1(x)))
#         x = self.relu2(self.gn2(self.cls_c2(x)))
#         x = self.relu3(self.gn3(self.cls_c3(x)))
#         x = self.relu4(self.gn4(self.cls_c4(x)))
#         cls = self.cls(x)
#         # center = self.center(x)
#         return cls
#
#
# class RegressionSub(nn.Module):
#     def __init__(self, feature):
#         super(RegressionSub, self).__init__()
#         self.reg_c1 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
#         self.reg_c2 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
#         self.reg_c3 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
#         self.reg_c4 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
#         self.reg = nn.Conv2d(feature, 4, kernel_size=3, padding=1)
#         self.center = nn.Conv2d(feature, 1, kernel_size=3, padding=1)
#         self.gn = nn.GroupNorm(32, feature)
#         self.gn2 = nn.GroupNorm(32, feature)
#         self.gn3 = nn.GroupNorm(32, feature)
#         self.gn4 = nn.GroupNorm(32, feature)
#         self.relu = nn.ReLU(inplace=True)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.relu4 = nn.ReLU(inplace=True)
#         self.apply(init_conv_rand_nomal)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.relu(self.gn(self.reg_c1(x)))
#         x = self.relu2(self.gn2(self.reg_c2(x)))
#         x = self.relu3(self.gn3(self.reg_c3(x)))
#         x = self.relu4(self.gn4(self.reg_c4(x)))
#         reg = self.reg(x)
#         center = self.center(x)
#         return center, reg
#


class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)


def init_conv_rand_nomal(module, std: int = 0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class GenTargets(nn.Module):
    def __init__(self, strides: List[int], limit_range: List[int]):
        super(GenTargets, self).__init__()
        self.stride = strides
        self.lim_range = limit_range
        assert len(strides) == len(limit_range)  # s와 범위 동일해야됨

    def forward(self, x):
        cls_logit, center_logit, reg_logit = x[0]
        gt_box = x[1]
        labels = x[2]
        cls_target = []
        center_target = []
        reg_target = []
        assert len(self.stride) == len(cls_logit)
        for lv in range(len(cls_logit)):
            lv_out = [cls_logit[lv], center_logit[lv], reg_logit[lv]]
            level_targets = self.generate_target(lv_out, gt_box, labels, self.stride[lv], self.lim_range[lv])
            cls_target.append(level_targets[0])
            center_target.append(level_targets[1])
            reg_target.append(level_targets[2])
        return torch.cat(cls_target, dim=1), torch.cat(center_target, dim=1), torch.cat(reg_target, dim=1)

    def generate_target(self, lv_out, gt_box, labels, stride, lim_range, sample_radio_ratio=1.5):

        cls_logit, center_logit, reg_logit = lv_out
        batch = cls_logit.shape[0]
        class_num = cls_logit.shape[1]
        # m = gt_box.shape[1]

        cls_logit = cls_logit.permute(0, 2, 3, 1)  # b,n,h,w -> b,h,w,c
        coords = coords_origin_fcos(feature = cls_logit, strides = stride).to(cls_logit.device)

        cls_logit = cls_logit.reshape((batch, -1, class_num))
        # center_logit = center_logit.permute(0, 2, 3, 1)
        # center_logit = center_logit.reshape((batch, -1, 1))
        # reg_logit = reg_logit.permute(0, 2, 3, 1)
        # reg_logit = reg_logit.reshape((batch, -1, 4))

        hw = cls_logit.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]
        left_offset = x[None, :, None] - gt_box[..., 0][:, None, :]  # ... 생략 객체
        top_offset = y[None, :, None] - gt_box[..., 1][:, None, :]
        right_offset = gt_box[..., 2][:, None, :] - x[None, :, None]
        bottom_offset = gt_box[..., 3][:, None, :] - y[None, :, None]
        offset = torch.stack([left_offset, top_offset, right_offset, bottom_offset], dim=-1)

        area = (offset[..., 0]+offset[..., 2]) * (offset[..., 1]+offset[..., 3])  # h * w = size

        offset_min = torch.min(offset, dim=-1)[0]
        offset_max = torch.max(offset, dim=-1)[0]

        mask_gt = offset_min > 0
        mask_lv = (offset_max > lim_range[0]) & (offset_max <= lim_range[1])

        ratio = stride * sample_radio_ratio
        gt_center_x = (gt_box[..., 0] + gt_box[..., 2]) / 2
        gt_center_y = (gt_box[..., 1] + gt_box[..., 3]) / 2
        gt_left_offset = x[None, :, None] - gt_center_x[:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        gt_top_offset = y[None, :, None] - gt_center_y[:, None, :]
        gt_right_offset = gt_center_x[:, None, :] - x[None, :, None]
        gt_bottom_offset = gt_center_y[:, None, :] - y[None, :, None]
        gt_offset = torch.stack([gt_left_offset, gt_top_offset, gt_right_offset, gt_bottom_offset], dim =-1)
        gt_off_max = torch.max(gt_offset, dim = -1)[0]
        mask_center = gt_off_max < ratio

        mask_pos = mask_gt & mask_lv & mask_center  #

        area[~mask_pos] = 99999999
        area_min_index = torch.min(area, dim = -1)[1]
        reg_target = offset[torch.zeros_like(area, dtype = torch.bool).scatter_(-1, area_min_index.unsqueeze(dim=-1), 1)]
        reg_target = torch.reshape(reg_target, (batch, -1, 4))

        classes = torch.broadcast_tensors(labels[:, None, :], area.long())[0]
        cls_target = classes[torch.zeros_like(area, dtype = torch.bool).scatter_(-1, area_min_index.unsqueeze(dim= -1), 1)]
        cls_target = torch.reshape(cls_target, (batch, -1, 1))

        left_right_min = torch.min(reg_target[..., 0], reg_target[..., 2])  # [batch_size,h*w]
        left_right_max = torch.max(reg_target[..., 0], reg_target[..., 2])
        top_bottom_min = torch.min(reg_target[..., 1], reg_target[..., 3])
        top_bottom_max = torch.max(reg_target[..., 1], reg_target[..., 3])
        center_target = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim=-1)

        assert reg_target.shape == (batch, hw, 4)
        assert cls_target.shape == (batch, hw, 1)
        assert center_target.shape == (batch, hw, 1)

        mask_pos_2 = mask_pos.long().sum(dim = -1)
        mask_pos_2 = mask_pos_2 >= 1
        assert mask_pos_2.shape == (batch, hw)
        cls_target[~mask_pos_2] = 0
        center_target[~mask_pos_2] = -1
        reg_target[~mask_pos_2] = -1

        return cls_target, center_target, reg_target


class Loss(nn.Module):
    def __init__(self, mode: str = 'giou'):
        super(Loss, self).__init__()
        self.mode = mode

    def forward(self, input):
        pred, target = input
        cls_logit, cen_logit, reg_logit = pred
        cls_target, cen_target, reg_target = target
        mask_pos = (cen_target > -1).squeeze(dim=-1)  ## sqeeze dim 1 제거
        cls_loss = self.compute_cls_loss(cls_logit, cls_target, mask_pos).mean()  #평균 이유 : 배치 당 로스
        cnt_loss = self.compute_cnt_loss(cen_logit, cen_target, mask_pos).mean()
        reg_loss = self.compute_reg_loss(reg_logit, reg_target, mask_pos, self.mode).mean()
        total_loss = cls_loss + cnt_loss + reg_loss
        return cls_loss, cnt_loss, reg_loss, total_loss

    def compute_cls_loss(self, preds, target, mask):
        batch_size = target.shape[0]
        preds_reshape = []
        class_num = preds[0].shape[1] # 20 ?
        mask = mask.unsqueeze(dim = -1)   #  torch.Size([2, 4724, 1])
        # mask=targets>-1#[batch_size,sum(_h*_w),1]
        num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # [batch_size,]
        for pred in preds:
            pred = pred.permute(0, 2, 3, 1)
            pred = torch.reshape(pred, [batch_size, -1, class_num])
            preds_reshape.append(pred)
        preds = torch.cat(preds_reshape, dim=1)  # [batch_size,sum(_h*_w),class_num]
        assert preds.shape[:2] == target.shape[:2]
        loss = []
        for batch_index in range(batch_size):
            pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
            target_pos = target[batch_index]  # [sum(_h*_w),1]
            target_pos = (torch.arange(1, class_num + 1, device = target_pos.device)[None, :] == target_pos).float()
            # sparse--> one-hot
            loss.append(self.focal_loss_from_logits(pred_pos, target_pos).view(1))
        return torch.cat(loss, dim=0) / num_pos  # [batch_size,]

    def compute_cnt_loss(self, preds, target, mask):
        batch_size = target.shape[0]
        c = target.shape[-1]
        preds_reshape = []
        mask = mask.unsqueeze(dim = -1)
        # mask=targets>-1#[batch_size,sum(_h*_w),1]
        num_pos = torch.sum(mask, dim = [1, 2]).clamp_(min = 1).float()  # [batch_size,]
        for pred in preds:
            pred = pred.permute(0, 2, 3, 1)
            pred = torch.reshape(pred, [batch_size, -1, c])
            preds_reshape.append(pred)
        preds = torch.cat(preds_reshape, dim = 1) ## 차원 증가
        assert preds.shape == target.shape  # [batch_size,sum(_h*_w),1]
        loss = []
        for batch_index in range(batch_size):
            pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,]
            target_pos = target[batch_index][mask[batch_index]]  # [num_pos_b,]
            assert len(pred_pos.shape) == 1
            loss.append(nn.functional.binary_cross_entropy_with_logits(input = pred_pos,
                                                           target = target_pos,
                                                           reduction = 'sum').view(1))
            return torch.cat(loss, dim = 0) / num_pos  # [batch_size,]

    def compute_reg_loss(self, preds, target, mask, mode='iou'):
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
        loss = []
        for batch_index in range(batch_size):
            pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,4]
            target_pos = target[batch_index][mask[batch_index]]  # [num_pos_b,4]
            assert len(pred_pos.shape) == 2
            if mode == 'iou':
                loss.append(self.iou_loss(pred_pos, target_pos).view(1))
            elif mode == 'giou':
                loss.append(self.giou_loss(pred_pos, target_pos).view(1))
            else:
                raise NotImplementedError("reg loss only implemented ['iou','giou']")
        return torch.cat(loss, dim = 0) / num_pos  # [batch_size,]

    def iou_loss(self, preds, targets):

        '''
        Args:
        preds: [n,4] ltrb
        targets: [n,4]
        '''

        lt = torch.min(preds[:, :2], targets[:, :2])
        rb = torch.min(preds[:, 2:], targets[:, 2:])
        wh = (rb + lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]  # [n]
        area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
        area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
        iou = overlap / (area1 + area2 - overlap)
        loss = -iou.clamp(min=1e-6).log()
        return loss.sum()

    def focal_loss_from_logits(self, preds, targets, gamma=2.0, alpha=0.25):
        '''
        Args:
        preds: [n,class_num]
        targets: [n,class_num]
        '''
        preds = preds.sigmoid()
        pt = preds * targets + (1.0 - preds) * (1.0 - targets)
        w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = -w * torch.pow((1.0 - pt), gamma) * pt.log()
        return loss.sum()

    def giou_loss(self, preds, targets):
        '''
        Args:
        preds: [n,4] ltrb
        targets: [n,4]
        '''
        lt_min = torch.min(preds[:, :2], targets[:, :2])
        rb_min = torch.min(preds[:, 2:], targets[:, 2:])
        wh_min = (rb_min + lt_min).clamp(min=0)
        overlap = wh_min[:, 0] * wh_min[:, 1]  # [n]
        area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
        area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
        union = (area1 + area2 - overlap)
        iou = overlap / union

        lt_max = torch.max(preds[:, :2], targets[:, :2])
        rb_max = torch.max(preds[:, 2:], targets[:, 2:])
        wh_max = (rb_max + lt_max).clamp(0)
        G_area = wh_max[:, 0] * wh_max[:, 1]  # [n]

        giou = iou - (G_area - union) / G_area.clamp(1e-10)
        loss = 1. - giou
        return loss.sum()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCOS(in_channel=[2048,1024,512], num_class = 20, feature=256).to(device)
    # a = torch.rand(1,3,512, 512).to(device)
    # tns = torch.rand(1, 3, 512, 512).to(device)
    model_info(model, 1, 3, 512, 512, device)
    # from torch.utils.tensorboard import SummaryWriter
    # import os
    #
    # writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', 'fcos'))
    #
    # writer.add_graph(model, tns)
    # writer.close()
