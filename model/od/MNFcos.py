import numpy as np
import torch
import torch.nn as nn
from utill.utills import model_info
from model.backbone.resnet50 import ResNet50v2
from model.modules.modules import MNBlock, ScaleExp
from typing import List


class MNFCOS(nn.Module):
    def __init__(self, in_channel: List[int], num_class: int, feature: int, freeze_bn: bool = True):
        super(MNFCOS, self).__init__()
        self.backbone = ResNet50v2()
        self.FeaturePyramidNetwork = LieghtWeightFeaturePyramid(in_channel, feature)
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


class LieghtWeightFeaturePyramid(nn.Module):
    def __init__(self, in_channel: List[int], feature=256):
        super(LieghtWeightFeaturePyramid, self).__init__()
        self.C5DownSample = nn.MaxPool2d(2, 2)
        self.C5UpSample = nn.Upsample(scale_factor=2)
        self.C5PW1 = nn.Conv2d(in_channels=in_channel[0], out_channels=feature, kernel_size=1)  # BN ACT 고민하
        self.C5PW2 = nn.Conv2d(in_channels=in_channel[0], out_channels=feature, kernel_size=1)
        self.C4PW = nn.Conv2d(in_channels=in_channel[1], out_channels = feature, kernel_size = 1)
        self.C3DownSample = nn.MaxPool2d(2, 2)
        self.C3PW = nn.Conv2d(in_channels = in_channel[2], out_channels = feature, kernel_size = 1)
        self.BaseFeature = nn.Conv2d(feature*3, feature, 1, bias=False)
        self.BaseFeatureBN = nn.BatchNorm2d(feature)
        self.BaseFeatureACT = nn.SiLU(True)
        self.BaseFeatureUpSample = nn.Upsample(scale_factor=2)
        self.BaseFeatureDownSample = nn.MaxPool2d(2, 2)
        # self.C3MNblock = MNBlock(feature, feature, 3, 3)
        self.P3MNblock = MNBlock(feature, feature, 3, 2)
        self.P4MNblock = MNBlock(feature, feature, 3, 2)
        self.P4MNblockDownSample = nn.MaxPool2d(2, 2)
        self.P5MNblock = MNBlock(feature, feature, 3, 2)
        self.P5MNblockDownSample = nn.MaxPool2d(2, 2)
        self.P6MNblock = MNBlock(feature, feature, 3, 2)
        self.P6MNblockDownSample = nn.MaxPool2d(2, 2)
        self.P7MNblock = MNBlock(feature, feature, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3, c4, c5 = x
        c5_top = self.C5DownSample(c5) # 8
        c5_top = self.C5PW1(c5_top)
        c5 = self.C5UpSample(c5) # 32
        c5 = self.C5PW2(c5) # 32
        c4 = self.C4PW(c4) # 32
        c3 = self.C3DownSample(c3) # 32
        c3 = self.C3PW(c3) #32
        # c3_MN = self.C3MNblock(c3)
        base_feature = torch.concat([c5, c4, c3], dim=1)  # 32
        base_feature = self.BaseFeature(base_feature)
        base_feature = self.BaseFeatureBN(base_feature)
        base_feature = self.BaseFeatureACT(base_feature)
        base_feature_top = self.BaseFeatureUpSample(base_feature)

        p3 = self.P3MNblock(base_feature_top)  # 64
        p4 = self.P4MNblock(base_feature)  # 32
        p5 = self.P4MNblockDownSample(p4)  # 16
        p5 = self.P5MNblock(p5)
        p6 = self.P5MNblockDownSample(p5)  # 8
        p6 = torch.add(p6, c5_top)
        p6 = self.P6MNblock(p6)
        p7 = self.P6MNblockDownSample(p6)
        p7 = self.P7MNblock(p7)
        return p3, p4, p5, p6, p7


class MNHeadFCOS(nn.Module):
    def __init__(self, feature: int, num_class: int, prior: float = 0.01):
        super(MNHeadFCOS, self).__init__()
        self.class_num = num_class
        self.prior = prior
        self.block1 = MNBlock(feature, feature, 3, 2)
        cls_branch = []
        reg_branch = []

        cls_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False))
        cls_branch.append(nn.BatchNorm2d(feature))
        cls_branch.append(nn.ReLU(True))

        reg_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding=1, bias=False))
        reg_branch.append(nn.BatchNorm2d(feature))
        reg_branch.append(nn.ReLU(True))

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(feature, self.class_num, kernel_size=3, padding=1)
        self.cnt_logits = nn.Conv2d(feature, 1, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(feature, 4, kernel_size=3, padding=1)

        nn.init.constant_(self.cls_logits.bias, -np.log((1 - self.prior) / self.prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        for index, P in enumerate(inputs):
            P = self.block1(P)
            cls_conv_out = self.cls_conv(P)
            reg_conv_out = self.reg_conv(P)
            cls_logits.append(self.cls_logits(cls_conv_out))
            cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNFCOS(in_channel=[2048, 1024, 512], num_class = 20, feature=128).to(device) # Total GFLOPs: 102.2868
    tns = torch.rand(1, 3, 512, 512).to(device)
    model_info(model, 1, 3, 512, 512, device)  # flop51.26G  para0.03G