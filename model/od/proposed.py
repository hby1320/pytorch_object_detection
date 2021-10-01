import torch
import torch.nn as nn

from model.modules.modules import PointWiseConv, DepthWiseConv2d
from model.backbone.resnet50 import ResNet50
from utill.utills import model_info
from typing import List
import numpy as np
import math


class FRFCOS(nn.Module):
    """
    Total params: 33,157,276
    Trainable params: 32,862,812
    Non-trainable params: 294,464
    Total mult-adds (G): 39.17
    Input size (MB): 3.154
    Forward/backward pass size (MB): 1087.0
    Params size (MB): 132.63
    """
    def __init__(self, feature_map: List[int],
                 feature_lv: List[int],
                 num_classes: int,
                 feature: int,
                 bn_freeze: bool = True):
        super(FRFCOS, self).__init__()
        self.backbone = ResNet50(3)
        self.backbone_freeze = bn_freeze
        self.fpn = ICSPFPN(feature_map, feature_lv)
        self.refine = RefineModule(feature_lv, 256)
        self.head = HeadFRFCOS(feature, num_classes, 0.01)
        self.act1 = nn.ReLU(True)
        self.act2 = nn.ReLU(True)
        self.act3= nn.ReLU(True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters():
                    p.requires_grad = False

        if self.backbone_freeze:
            self.apply(freeze_bn)
            self.backbone.freeze_stages(1)
            print(f"success frozen BN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = self.backbone(x)  # 512 * 64, 1024 * 32, 2048  * 16
        x = self.fpn([x1, x2, x3])
        x = self.refine(x)
        cls, cnt, reg = self.head(x)
        return cls, cnt, reg


class ICSPFPN(nn.Module):
    def __init__(self, feature_map: List[int], feature_lv: List[int]):
        super(ICSPFPN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=feature_map[2], out_channels=feature_map[1], kernel_size=1,
                               stride=1, bias=False)
        self.bn = nn.BatchNorm2d(feature_map[1])
        self.act = nn.ReLU(True)
        self.tf1 = PointWiseConv(feature_map[1],feature_lv[2], bs= True)
        self.tf2 = PointWiseConv(feature_map[0],feature_lv[1], bs= True)
        self.icsp_blcok1 = ICSPBlock(feature_map[1], feature_lv[2])
        self.icsp_blcok2 = ICSPBlock(feature_lv[2], feature_lv[1])
        self.icsp_blcok3 = ICSPBlock(feature_lv[1], feature_lv[0])
        self.Up_sample1 = nn.Upsample(scale_factor = 2)
        self.Up_sample2 = nn.Upsample(scale_factor = 2)
    # [512, 1024, 2048], [128, 256, 512]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x
        x3 = self.act(self.bn(self.conv1(x3)))  # 1024 16 16
        p5 = self.icsp_blcok1(x3)  # 512 16 16  @
        p5_1 = self.Up_sample1(p5)  # 512 32 32
        x2 = self.tf1(x2)  # 512 32 32
        p5_1 = torch.add(p5_1, x2)  # 512 32 32
        p6 = self.icsp_blcok2(p5_1)  # 256 32 32   @
        p6_1 = self.Up_sample2(p6)   # 256 64 64
        x1 = self.tf2(x1)  # 256 64 64
        p6_1 = torch.add(p6_1, x1)  # 256 64 64
        p7 = self.icsp_blcok3(p6_1)  #128 64 64 @
        return p5, p6, p7

class ICSPBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, beta:int =2, alpha:int = 4):
        super(ICSPBlock, self).__init__()

        self.pw_conv1 = PointWiseConv(in_channel=in_ch, out_channel=in_ch//2)
        self.pw_conv2 = PointWiseConv(in_channel=in_ch//2, out_channel=in_ch)
        self.pw_conv3 = PointWiseConv(in_channel=in_ch, out_channel=in_ch//2)
        self.pw_conv4 = PointWiseConv(in_channel=in_ch//2, out_channel=in_ch//2, bs=True)
        self.pw_conv5 = PointWiseConv(in_channel=in_ch, out_channel=in_ch//2)
        self.dw_conv1 = DepthWiseConv2d(in_ch, 3, 1)
        self.se_block = SEBlock(in_ch, alpha=alpha)
        self.bn = nn.BatchNorm2d(in_ch//2)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.bn3 = nn.BatchNorm2d(in_ch//2)
        self.bn4 = nn.BatchNorm2d(in_ch//2)
        self.act = nn.ReLU(True)
        self.act1 = nn.ReLU(True)
        self.act2 = nn.ReLU(True)
        self.act3 = nn.ReLU(True)
        self.act4 = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [1, 512, 8, 8]
        x1 = self.act(self.bn(self.pw_conv1(x)))
        x2 = self.act1(self.bn1(self.pw_conv2(x1)))
        x2 = self.act2(self.bn2(self.dw_conv1(x2)))
        x2 = self.se_block(x2)
        x2 = self.act3(self.bn3(self.pw_conv3(x2)))
        x1 = self.pw_conv4(x1)
        x3 = torch.cat([x2,x1],dim=1)
        x3 = self.act4(self.bn4(self.pw_conv5(x3)))
        return x3


class SEBlock(nn.Module):
    def __init__(self, feature:int,  alpha:int = 4):
        super(SEBlock, self).__init__()
        self.Gap = nn.AdaptiveAvgPool2d(1)
        self.pw_conv1 = PointWiseConv(in_channel=feature, out_channel=feature // 4, bs=True)
        self.act = nn.ReLU()
        self.pw_conv2 = PointWiseConv(in_channel=feature // 4, out_channel=feature, bs=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.Gap(x)  # torch.Size([1, 512, 1, 1])
        x1 = self.pw_conv1(x1)
        x1 = self.act(x1)
        x1 = self.pw_conv2(x1)
        x1 = self.sigmoid(x1)
        return x1 * x


class RefineModule(nn.Module):
    def __init__(self, feature_lv: List[int], feature:int):
        super(RefineModule, self).__init__()
        self.up_sample1 = nn.Upsample(scale_factor=2)
        self.up_sample2 = nn.Upsample(scale_factor=2)
        self.down_sample1 = nn.MaxPool2d(2, 2)
        self.down_sample2 = nn.MaxPool2d(2, 2)
        self.pw_conv1 = PointWiseConv(feature_lv[2] + feature_lv[1], feature, bs=True)
        self.pw_conv2 = PointWiseConv(feature_lv[1] + feature_lv[0], feature, bs=True)
        self.pw_conv3 = nn.Conv2d(feature, feature, 3, 1, 1)
        self.pw_conv4 = PointWiseConv(feature, feature_lv[2])
        self.pw_conv5 = PointWiseConv(feature, feature_lv[1])
        self.pw_conv6 = PointWiseConv(feature, feature_lv[0])
        self.tf1 = PointWiseConv(feature_lv[2], feature)
        self.tf2 = PointWiseConv(feature_lv[1], feature)
        self.tf3 = PointWiseConv(feature_lv[0], feature)
        self.act = nn.ReLU(True)
        self.act1 = nn.ReLU(True)
        self.act2 = nn.ReLU(True)
        self.act3 = nn.ReLU(True)
        self.bn = nn.BatchNorm2d(feature)
        self.bn1 = nn.BatchNorm2d(feature_lv[2])
        self.bn2 = nn.BatchNorm2d(feature_lv[1])
        self.bn3 = nn.BatchNorm2d(feature_lv[0])

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x  # hihg -> low ch
        x1_up = self.up_sample1(x1)
        x = torch.cat([x1_up, x2], dim=1)
        x = self.pw_conv1(x)
        x3_down = self.down_sample1(x3)
        x = torch.cat([x, x3_down], dim=1)
        x = self.pw_conv2(x)

        # Refine
        x = self.act(self.bn(self.pw_conv3(x)))
        x_up = self.up_sample1(x)
        x_down = self.down_sample2(x)
        x_down = self.act1(self.bn1(self.pw_conv4(x_down)))
        x = self.act2(self.bn2(self.pw_conv5(x)))
        x_up = self.act3(self.bn3(self.pw_conv6(x_up)))
        x1 = self.tf1(torch.add(x_down, x1))  # 512 16
        x2 = self.tf2(torch.add(x, x2))  # 256 32
        x3 = self.tf3(torch.add(x_up, x3))  # 128 64
        return x1, x2, x3


class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)


class HeadFRFCOS(nn.Module):
    def __init__(self, feature:int, num_class: int, prior: float = 0.01):
        super(HeadFRFCOS, self).__init__()
        self.class_num = num_class
        self.prior = prior
        cls_branch = []
        reg_branch = []

        for i in range(4):
            cls_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False))
            # reg_branch.append(nn.GroupNorm(32, feature))
            cls_branch.append(nn.BatchNorm2d(feature))
            cls_branch.append(nn.ReLU(True))
            reg_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False))
            # reg_branch.append(nn.GroupNorm(32, feature))
            reg_branch.append(nn.BatchNorm2d(feature))
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FRFCOS([512, 1024, 2048], [128, 256, 512], 20, 256).to(device)
    model_info(model, 1, 3, 512, 512, device)
    # tns = torch.rand(1, 3, 512, 512).to(device)

    # from torch.utils.tensorboard import SummaryWriter
    # import os
    # writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', 'proposed'))
    # #
    # writer.add_graph(model, tns)
    # writer.close()

