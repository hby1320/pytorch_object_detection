import numpy as np
import torch
import torch.nn as nn
from utill.utills import model_info
from model.backbone.resnet50 import ResNet50v2
from model.modules.modules import MNBlock, SEBlock, DeformableConv2d, ScaleExp
from typing import List
from model.backbone.efficientnetv1 import EfficientNetV1


class MNFCOS(nn.Module):
    def __init__(self, in_channel: List[int], num_class: int, feature: int, freeze_bn: bool = True):
        super(MNFCOS, self).__init__()
        self.backbone = ResNet50v2()
        self.FeaturePyramidNetwork = LieghtWeightFeaturePyramid(in_channel, feature)
        # self.FeaturePyramidNetwork = LieghtWeightFeaturePyramid_old(in_channel, feature)
        self.head = MNHeadFCOS(feature, num_class, 0.01)
        self.backbone_freeze = freeze_bn

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters():
                    p.requires_grad = False  # 학습 x
        if self.backbone_freeze:
            self.apply(freeze_bn)
            print(f"success frozen BN")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        x = self.FeaturePyramidNetwork(x)
        cls, cnt, reg = self.head(x)
        return cls, cnt, reg


class LieghtWeightFeaturePyramid(nn.Module):
    def __init__(self, in_channel: List[int], feature=128):
        super(LieghtWeightFeaturePyramid, self).__init__()
        self.C5PW = nn.Conv2d(in_channels=in_channel[0], out_channels= feature, kernel_size=1)
        self.C4PW = nn.Conv2d(in_channels=in_channel[1], out_channels = feature, kernel_size = 1)
        self.C3PW = nn.Conv2d(in_channels = in_channel[2], out_channels = feature, kernel_size = 1)
        self.C3_Deform1 = DeformableConv2d(in_channel[1], in_channel[1], kernel_size=3, padding=1, bias=True)
        self.C3DownSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C5UpSample = nn.Upsample(scale_factor=2)
        # backbone channel tune
        self.BaseFeatureUpSample = nn.Upsample(scale_factor=2)
        self.BaseFeatureDownSample = nn.MaxPool2d(2, 2)
        self.Base = nn.Conv2d(feature * 3, feature, 1, 1, bias=True)

        self.P3MNblock = MNBlock(feature, feature, 3, 2, 3)
        self.P4MNblock = MNBlock(feature, feature, 3, 2, 3)
        self.P4MNblockDownSample = nn.MaxPool2d(2, stride=2)
        self.P5MNblock = MNBlock(feature, feature, 3, 2, 2)
        self.P5MNblockDownSample = nn.MaxPool2d(2, stride=2)
        self.P6MNblock = MNBlock(feature, feature, 3, 2, 2)

        self.Base_pw1 = nn.Conv2d(feature * 3, feature * 2, 1, 1, bias=True)
        self.Base_dw1_1 = nn.Conv2d(feature * 2, feature * 2, 5, 1, 5 // 2, 1, feature * 2, bias=True)
        self.Base_dw1_2 = nn.Conv2d(feature * 2, feature * 2, 7, 1, 7 // 2, 1, feature * 2, bias=True)
        self.Base_dw1_3 = nn.Conv2d(feature * 2, feature * 2, 3, 1, 3 // 2, 1, feature * 2, bias=False)
        self.sig1 = nn.Sigmoid()
        self.Base_bn1 = nn.BatchNorm2d(feature * 2)
        self.Base_act1 = nn.SiLU(True)
        self.Base_dw1_4 = nn.Conv2d(feature * 3, feature * 3, 3, 1, 3 // 2, 1, feature * 3, bias=False)
        self.Base_bn2 = nn.BatchNorm2d(feature * 3)
        self.Base_act2 = nn.SiLU(True)
        self.Base_pw2 = nn.Conv2d(feature * 2, feature, 1, 1, bias=True)
        # self.Base_pw3 = nn.Conv2d(feature, feature, 1, 1, bias=True)

        self.Base_down = nn.MaxPool2d(2, 2)
        self.Base_up = nn.Upsample(scale_factor=2)
        self.DW_1 = nn.Conv2d(feature, feature, 3, 1, 3 // 2, 1, feature, bias=False)
        self.DW_1_bn1 = nn.BatchNorm2d(feature)
        self.DW_1_act1 = nn.SiLU(True)

        self.DW_2 = nn.Conv2d(feature, feature, 3, 1, 3 // 2, 1, feature, bias=False)
        self.DW_2_BN = nn.BatchNorm2d(feature)
        self.DW_2_ACT = nn.SiLU(True)

        self.DW_3 = nn.Conv2d(feature, feature, 3, 1, 3 // 2, 1, feature, bias=False)
        self.DW_3_BN = nn.BatchNorm2d(feature)
        self.DW_3_ACT = nn.SiLU(True)
        self.SE1 = SEBlock(feature * 3, 4)

        self.P3_2MNblock = MNBlock(feature, feature, 3, 2, 1)
        self.P4_2MNblock = MNBlock(feature, feature, 3, 2, 1)
        self.P5_2MNblock = MNBlock(feature, feature, 3, 2, 1)
        self.P6_2MNblock = MNBlock(feature, feature, 3, 2, 1)
        # self.P5_2MNblock = MNBlock(feature, feature, 3, 2, 2)
        # self.P6_2MNblockUpwnSample = nn.Upsample(scale_factor=2)
        # self.P4_2MNblock = MNBlock(feature, feature, 3, 2, 2)
        # self.P5_2MNblockUpwnSample = nn.Upsample(scale_factor=2)
        # self.P3_2MNblock = MNBlock(feature, feature, 3, 2, 2)
        # self.P4_2MNblockUpwnSample = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = x
        c3 = self.C3PW(c3)  # 256 64 64
        c3_2 = self.C3DownSample(c3)  # c3 -> 256 32 32
        c4 = self.C3_Deform1(c4)
        c4 = self.C4PW(c4)  # 256 32 32
        c5 = self.C5PW(c5)  # 256 16 16
        c5_2 = self.C5UpSample(c5)  # C5 -> 256 32 32


        base_feature = torch.concat([c5_2, c4, c3_2], dim=1)  # 384 32 32
        base_feature = self.SE1(base_feature)
        base_feature = self.Base_pw1(base_feature)
        base_1 = self.Base_dw1_1(base_feature)
        base_2 = self.Base_dw1_2(base_feature)
        base_3 = torch.add(base_1, base_2)
        # base_3 = self.Base_pw2(base_3)
        base_3 = self.Base_dw1_3(base_3)
        base_3 = self.Base_bn1(base_3)
        base_3 = self.Base_act1(base_3)
        base_3 = torch.mul(self.sig1(base_3), base_feature)
        base = self.Base_pw2(base_3)

        # base_base = self.Base_pw3(base_l)
        # base_1 = self.Base_dw2_1(base_base)
        # base_2 = self.Base_dw2_2(base_base)
        # base_3 = torch.add(base_1, base_2)
        # base_3 = self.Base_dw2_3(base_3)
        # base_3 = self.Base_bn2(base_3)
        # base_3 = self.Base_act2(base_3)
        # base_3 = torch.mul(self.sig2(base_3), base_base)
        # base_m = self.Base_pw4(base_3)
        #
        # base_base2 = self.Base_pw5(base_m)
        # base_1 = self.Base_dw3_1(base_base2)
        # base_2 = self.Base_dw3_2(base_base2)
        # base_3 = torch.add(base_1, base_2)
        # base_3 = self.Base_dw3_3(base_3)
        # base_3 = self.Base_bn3(base_3)
        # base_3 = self.Base_act3(base_3)
        # base_3 = torch.mul(self.sig3(base_3), base_base2)
        # base_s = self.Base_pw6(base_3)

        base = self.DW_1(base)
        base = self.DW_1_bn1(base)
        base = self.DW_1_act1(base)

        base_s = self.Base_down(base)  # 64

        base_s = self.DW_2(base_s)  # 16
        base_s = self.DW_2_BN(base_s)  # 16
        base_s = self.DW_2_ACT(base_s)  # 16

        base_m = self.Base_up(base)  # 64

        # base_m = torch.add(c3, base_m)
        base_m = self.DW_3(base_m)
        base_m = self.DW_3_BN(base_m)
        base_m = self.DW_3_ACT(base_m)
        p4 = self.P4MNblock(base)  # 32A
        p4 = self.P4_2MNblock(p4)  # 32A #
        p3 = self.BaseFeatureUpSample(base)
        p5 = self.P4MNblockDownSample(p4)  # 16
        p3 = torch.add(p3, base_m)
        p3 = self.P3MNblock(p3)  # 64
        p3 = self.P3_2MNblock(p3)  # 64
        p5 = torch.add(p5, base_s)

        p5 = self.P5MNblock(p5)
        p5 = self.P5_2MNblock(p5)  #

        p6 = self.P5MNblockDownSample(p5)  # 8

        p6 = self.P6MNblock(p6)
        p6 = self.P6_2MNblock(p6)  #
        #  --
        # p6_2 = self.P6_2MNblockUpwnSample(p6)
        # p5 = torch.add(p6_2, p5)
        # p5 = self.P5_2MNblock(p5)
        # p5_2 = self.P5_2MNblockUpwnSample(p5)
        # p4 = torch.add(p5_2, p4)
        # p4 = self.P4_2MNblock(p4)
        # p4_2 = self.P4_2MNblockUpwnSample(p4)
        # p3 = torch.add(p3, p4_2)
        # p3 = self.P3_2MNblock(p3)

        return p3, p4, p5, p6


# class LieghtWeightFeaturePyramid_old(nn.Module):
#     def __init__(self, in_channel: List[int], feature=128):
#         super(LieghtWeightFeaturePyramid_old, self).__init__()
#         self.C5DownSample = nn.MaxPool2d(2, 2)
#         self.C5UpSample = nn.Upsample(scale_factor=2)
#         self.C5PW1 = nn.Conv2d(in_channels=in_channel[0], out_channels=feature, kernel_size=1)
#         self.C5PW2 = nn.Conv2d(in_channels=in_channel[0], out_channels=feature, kernel_size=1)
#         self.C4PW = nn.Conv2d(in_channels=in_channel[1], out_channels=feature, kernel_size=1)
#         self.C3DownSample = nn.MaxPool2d(2, 2)
#         self.C3PW = nn.Conv2d(in_channels=in_channel[2], out_channels=feature, kernel_size=1)
#         self.BaseFeature = nn.Conv2d(feature * 3, feature, 1, bias=False)
#         self.BaseFeatureBN = nn.BatchNorm2d(feature)
#         self.BaseFeatureACT = nn.SiLU(True)
#         self.BaseFeatureUpSample = nn.Upsample(scale_factor=2)
#         self.BaseFeatureDownSample = nn.MaxPool2d(2, 2)
#         # self.C3MNblock = MNBlock(feature, feature, 3, 3)
#         self.P3MNblock = MNBlock(feature, feature, 3, 3)
#         self.P4MNblock = MNBlock(feature, feature, 3, 3)
#         self.P4MNblockDownSample = nn.MaxPool2d(2, 2)
#         self.P5MNblock = MNBlock(feature, feature, 3, 3)
#         self.P5MNblockDownSample = nn.MaxPool2d(2, 2)
#         self.P6MNblock = MNBlock(feature, feature, 3, 2)
#         self.P6MNblockDownSample = nn.MaxPool2d(2, 2)
#         self.P7MNblock = MNBlock(feature, feature, 3, 2)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         c3, c4, c5 = x
#         c5_top = self.C5DownSample(c5)  # 8
#         c5_top = self.C5PW1(c5_top)
#         c5 = self.C5UpSample(c5)  # 32
#         c5 = self.C5PW2(c5)  # 32
#         c4 = self.C4PW(c4)  # 32
#         c3 = self.C3DownSample(c3)  # 32
#         c3 = self.C3PW(c3)  # 32
#         # c3_MN = self.C3MNblock(c3)
#         base_feature = torch.concat([c5, c4, c3], dim=1)  # 32
#         base_feature = self.BaseFeature(base_feature)
#         base_feature = self.BaseFeatureBN(base_feature)
#         base_feature = self.BaseFeatureACT(base_feature)
#         base_feature_top = self.BaseFeatureUpSample(base_feature)
#
#         p3 = self.P3MNblock(base_feature_top)  # 64
#         p4 = self.P4MNblock(base_feature)  # 32
#         p5 = self.P4MNblockDownSample(p4)  # 16
#         p5 = self.P5MNblock(p5)
#         p6 = self.P5MNblockDownSample(p5)  # 8
#         p6 = torch.add(p6, c5_top)
#         p6 = self.P6MNblock(p6)
#
#         return p3, p4, p5, p6

class MNHeadFCOS(nn.Module):
    def __init__(self, feature: int, num_class: int, prior: float = 0.01):
        super(MNHeadFCOS, self).__init__()
        self.class_num = num_class
        self.prior = prior
        self.block1 = MNBlock(feature, feature, 3, 2)
        cls_branch = []
        reg_branch = []
        cls_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False))
        cls_branch.append(nn.GroupNorm(32, feature))
        cls_branch.append(nn.SiLU(True))
        reg_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False))
        reg_branch.append(nn.GroupNorm(32, feature))
        reg_branch.append(nn.SiLU(True))
        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)
        # self.cls_logits = nn.Conv2d(feature, self.class_num, kernel_size=3, padding=1)
        # self.cnt_logits = nn.Conv2d(feature, 1, kernel_size=3, padding=1)
        # self.reg_pred = nn.Conv2d(feature, 4, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(feature, self.class_num, kernel_size=1)
        self.cnt_logits = nn.Conv2d(feature, 1, kernel_size=1)
        self.reg_pred = nn.Conv2d(feature, 4, kernel_size=1)
        nn.init.constant_(self.cls_logits.bias, -np.log((1 - self.prior) / self.prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(4)])

    def forward(self, inputs: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        for index, feature in enumerate(inputs):
            feature = self.block1(feature)
            cls_conv_out = self.cls_conv(feature)
            reg_conv_out = self.reg_conv(feature)
            cls_logits.append(self.cls_logits(cls_conv_out))
            cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNFCOS(in_channel=[2048, 1024, 512], num_class = 20, feature=128).to(device)
    tns = torch.rand(1, 3, 512, 512).to(device)
    model_info(model, 1, 3, 512, 512, device, 4)  # flop51.26G  para0.03G
