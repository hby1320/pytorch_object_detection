import numpy as np
import torch
import torch.nn as nn
from utill.utills import model_info
from model.backbone.resnet50 import ResNet50v2
from model.modules.modules import MNBlock, ScaleExp
from typing import List
# from model.backbone.efficientnetv1 import EfficientNetV1


class MNFCOS(nn.Module):
    def __init__(self, in_channel: List[int], num_class: int, feature: int, freeze_bn: bool = True):
        super(MNFCOS, self).__init__()
        self.backbone = ResNet50v2()
        # self.backbone = EfficientNetV1(0)
        self.FeaturePyramidNetwork = LightWeightFeaturePyramid(in_channel, feature)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.FeaturePyramidNetwork(x)
        cls, cnt, reg = self.head(x)
        return cls, cnt, reg


class LightWeightFeaturePyramid(nn.Module):
    def __init__(self, in_channel: List[int], feature=256):
        super(LightWeightFeaturePyramid, self).__init__()
        self.C5UpSample = nn.Upsample(scale_factor=2)
        self.C5PW1 = nn.Conv2d(in_channels=in_channel[0], out_channels=feature, kernel_size=1)  # BN ACT 고민하
        self.C5PW2 = nn.Conv2d(in_channels=in_channel[0], out_channels=feature, kernel_size=1)
        self.C4PW = nn.Conv2d(in_channels=in_channel[1], out_channels=feature, kernel_size=1)
        self.C3DownSample = nn.MaxPool2d(2, 2)
        self.C3PW = nn.Conv2d(in_channels=in_channel[2], out_channels=feature, kernel_size=1)
        self.BaseFeatureUpSample = nn.Upsample(scale_factor=2)
        self.BaseFeatureDownSample = nn.MaxPool2d(2, 2)
        self.Base = nn.Conv2d(feature * 3, feature, 1, 1, bias=True)
        self.BaseMNBlock = MNBlock(feature, feature, 3, 3)
        self.BaseMNBlock2 = MNBlock(feature, feature, 3, 3)

        self.C3MNBlock = MNBlock(feature, feature, 3, 3)
        self.P3MNBlock = MNBlock(feature, feature, 3, 3)
        self.P4MNBlock = MNBlock(feature, feature, 3, 3)
        self.P4MNBlockDownSample = nn.MaxPool2d(2, 2)
        self.P5MNBlock = MNBlock(feature, feature, 3, 3)
        self.P5MNBlockDownSample = nn.MaxPool2d(2, 2)
        self.P6MNBlock = MNBlock(feature, feature, 3, 3)

        self.Base_dw1 = nn.Conv2d(feature*3, feature*3, 5, 1, 5//2, 1, feature*3, bias=False)
        self.Base_act1 = nn.SiLU(True)
        self.Base_bn1 = nn.BatchNorm2d(feature*3)
        self.Base_pw1 = nn.Conv2d(feature*3, feature, 1, 1, bias=True)
        self.Base_pw2 = nn.Conv2d(feature, feature*3, 1, 1, bias=True)
        self.Base_dw2 = nn.Conv2d(feature*3, feature*3, 5, 1, 5//2, 1, feature*3, bias=False)
        self.Base_bn2 = nn.BatchNorm2d(feature * 3)
        self.Base_act2 = nn.SiLU(True)
        self.Base_pw3 = nn.Conv2d(feature*3, feature, 1, 1, bias=True)
        self.Base_pw4 = nn.Conv2d(feature, feature * 3, 1, 1, bias=True)
        self.Base_dw3 = nn.Conv2d(feature*3, feature*3, 5, 1, 5 // 2, 1, feature*3, bias=False)
        self.Base_bn3 = nn.BatchNorm2d(feature * 3)
        self.Base_act3 = nn.SiLU(True)
        self.Base_pw5 = nn.Conv2d(feature*3, feature, 1, 1, bias=True)
        self.Dw_3 = nn.Conv2d(feature, feature, 7, 1, 7//2, 1, feature, bias=False)
        self.Base_bn4 = nn.BatchNorm2d(feature)
        self.Base_act4 = nn.SiLU(True)
        self.Base_pw6 = nn.Conv2d(feature * 3, feature, 1, 1, bias=True)
        self.Base_down = nn.MaxPool2d(2, 2)
        self.Base_up = nn.Upsample(scale_factor=2)
        self.DW_1 = nn.Conv2d(feature, feature, 5, 1, 5//2, 1, feature, bias=True)
        self.DW_2 = nn.Conv2d(feature, feature, 5, 1, 5 // 2, 1, feature, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3, c4, c5 = x
        c5 = self.C5PW1(c5)  # 16
        c5_2 = self.C5UpSample(c5)  # 32
        c4 = self.C4PW(c4)  # 32
        c3 = self.C3PW(c3)  # 64
        c3_2 = self.C3DownSample(c3)  # 32

        base_feature = torch.concat([c5_2, c4, c3_2], dim=1)  # 256
        base_1 = self.Base_dw1(base_feature)
        base_1 = self.Base_bn1(base_1)
        base_1 = self.Base_act1(base_1)
        base_1 = self.Base_pw1(base_1)  # pw 384 -> 128
        base_feature = self.Base_pw6(base_feature)

        base_s = torch.add(base_1, base_feature)
        base_m = self.Base_pw2(base_s)  # feature*3
        base_m = self.Base_dw2(base_m)
        base_m = self.Base_bn2(base_m)
        base_m = self.Base_act2(base_m)
        base_m = self.Base_pw3(base_m)
        base_m = torch.add(base_m, base_s)
        base_l = self.Base_pw4(base_m)  # feature*3
        base_l = self.Base_dw3(base_l)
        base_l = self.Base_bn3(base_l)
        base_l = self.Base_act3(base_l)
        base_l = self.Base_pw5(base_l)
        base_l = torch.add(base_l, base_m)
        base_l = self.Dw_3(base_l)
        base_l = self.Base_bn4(base_l)
        base_l = self.Base_act4(base_l)
        base_s = self.DW_1(base_s)
        base_s = self.Base_down(base_s)
        base_m = self.DW_2(base_m)
        base_m = self.Base_up(base_m)

        p3 = self.BaseFeatureUpSample(base_l)
        p3 = torch.add(p3, base_m)
        p3 = self.P3MNBlock(p3)  # 64
        p4 = self.P4MNBlock(base_l)  # 32
        p5 = self.P4MNBlockDownSample(p4)  # 16
        p5 = torch.add(p5, base_s)
        p5 = self.P5MNBlock(p5)
        p6 = self.P5MNBlockDownSample(p5)  # 8
        p6 = self.P6MNBlock(p6)
        return p3, p4, p5, p6


class MNHeadFCOS(nn.Module):
    def __init__(self, feature: int, num_class: int, prior: float = 0.01):
        super(MNHeadFCOS, self).__init__()
        self.class_num = num_class
        self.prior = prior
        self.block1 = MNBlock(feature, feature, 3, 3)

        cls_branch = []
        reg_branch = []
        cls_branch.append(nn.Conv2d(feature, feature, kernel_size=5, padding=1, groups=feature, bias=False))
        cls_branch.append(nn.GroupNorm(32, feature))
        cls_branch.append(nn.SiLU(True))
        reg_branch.append(nn.Conv2d(feature, feature, kernel_size=5, padding=1, groups=feature, bias=False))
        reg_branch.append(nn.GroupNorm(32, feature))
        reg_branch.append(nn.SiLU(True))

        self.class_conv = nn.Sequential(*cls_branch)
        self.regression_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(feature, self.class_num, kernel_size=3, padding=1)
        self.cnt_logits = nn.Conv2d(feature, 1, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(feature, 4, kernel_size=3, padding=1)

        nn.init.constant_(self.cls_logits.bias, -np.log((1 - self.prior) / self.prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(4)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cls_logits = []
        cnt_logits = []
        reg_preds = []

        for index, feature in enumerate(inputs):
            feature = self.block1(feature)
            cls_conv_out = self.class_conv(feature)
            reg_conv_out = self.regression_conv(feature)
            cls_logits.append(self.cls_logits(cls_conv_out))
            cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNFCOS(in_channel=[2048, 1024, 512], num_class=20, feature=128).to(device)
    # model = MNFCOS(in_channel=[320, 112, 40], num_class = 20, feature=128).to(device) #
    tns = torch.rand(1, 3, 512, 512).to(device)
    model_info(model, 1, 3, 512, 512, device, 1)  # flop51.26G  para0.03G
