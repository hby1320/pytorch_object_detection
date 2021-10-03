import numpy as np
import torch
import torch.nn as nn
from utill.utills import model_info, coords_origin_fcos
from model.backbone.resnet50 import ResNet50

from typing import List


class FCOS(nn.Module):
    """
    Total params: 30,976,860
    Trainable params: 30,701,340
    Non-trainable params: 275,520
    Total mult-adds (G): 51.14

    Input size (MB): 3.15
    Forward/backward pass size (MB): 1086.27
    Params size (MB): 123.91
    Estimated Total Size (MB): 1213.32
    """
    def __init__(self, in_channel: List[int], num_class: int, feature: int, freeze_bn: bool = True):
        super(FCOS, self).__init__()
        self.backbone = ResNet50(3)
        self.FPN = FeaturePyramidNetwork(in_channel, feature)
        self.head = HeadFCOS(feature, num_class, 0.01)
        self.backbone_freeze = freeze_bn

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters():p.requires_grad = False  # 학습 x
        if self.backbone_freeze:
            self.apply(freeze_bn)
            print(f"success frozen BN")
            self.backbone.freeze_stages(1)
            print(f"success frozen_stage")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1,x2,x3 = self.backbone(x)
        x = self.FPN([x1,x2,x3])
        cls, cnt, reg = self.head(x)
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
        p6 = self.P6_c1(p5)
        p7 = self.P7_c1(self.act(p6))
        return [p3, p4, p5]


class HeadFCOS(nn.Module):
    def __init__(self, feature: int, num_class: int, prior: float = 0.01):
        super(HeadFCOS, self).__init__()
        self.class_num = num_class
        self.prior = prior
        cls_branch = []
        reg_branch = []

        for i in range(4):
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
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

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


class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)


class GenTargets(nn.Module):
    def __init__(self, strides: List[int], limit_range: List[List[int]]):
        super(GenTargets, self).__init__()
        self.stride = strides
        self.lim_range = limit_range
        assert len(strides) == len(limit_range)  # s와 범위 동일해야됨

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

    def generate_target(self,
                        lv_out : torch.Tensor,
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
        m = gt_box.shape[1]
        cls_logit = cls_logit.permute(0, 2, 3, 1)  # b,n,h,w -> b,h,w,c
        coords = coords_origin_fcos(feature = cls_logit, strides = stride).to(device=gt_box.device)  # [H*W , 2]
        cls_logit = cls_logit.reshape((batch, -1, class_num))  # [N, H*W, C]
        center_logit = center_logit.permute(0, 2, 3, 1)
        center_logit = center_logit.reshape((batch, -1, 1))
        reg_logit = reg_logit.permute(0, 2, 3, 1)
        reg_logit = reg_logit.reshape((batch, -1, 4))

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

        mask_gt = offset_min > 0  #음수일제 정답 아니기 때문에
        mask_lv = (offset_max > lim_range[0]) & (offset_max <= lim_range[1])  # 해당 특징맵 LV 이하 값 제거
        offset_min = torch.min(offset, dim=-1)[0] ## [0] minValue
        offset_max = torch.max(offset, dim=-1)[0] ## [0] Max

        mask_gt = offset_min > 0
        mask_lv = (offset_max > lim_range[0]) & (offset_max <= lim_range[1])

        ratio = stride * sample_radio_ratio
        gt_center_x = (gt_box[..., 0] + gt_box[..., 2]) / 2  # 중심 생성
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
        area_min_index = torch.min(area, dim = -1)[1]  # area minval index
        reg_target = offset[torch.zeros_like(area, dtype = torch.bool).scatter_(-1, area_min_index.unsqueeze(dim=-1), 1)]
        reg_target = torch.reshape(reg_target, (batch, -1, 4))

        labels = torch.broadcast_tensors(labels[:, None, :], area.long())[0]
        cls_target = labels[torch.zeros_like(area, dtype = torch.bool).scatter_(-1, area_min_index.unsqueeze(dim= -1), 1)]
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

        mask_pos_2 = mask_pos_2 >=1
        assert mask_pos_2.shape == (batch, hw)
        cls_target[~mask_pos_2] = 0
        center_target[~mask_pos_2] = -1
        reg_target[~mask_pos_2] = -1

        return cls_target, center_target, reg_target


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCOS(in_channel=[2048,1024,512], num_class = 20, feature=256).to(device)
    # a = torch.rand(1,3,512, 512).to(device)
    # tns = torch.rand(1, 3, 512, 512).to(device)
    model_info(model, 1, 3, 512, 512, device)  # flop51.26G  para0.03G
    # from torch.utils.tensorboard import SummaryWriter
    # # import os
    # #
    # # writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', 'fcos'))
    # #
    # # writer.add_graph(model, tns)
    # # writer.close()
