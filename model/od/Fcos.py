import numpy as np
import torch
import torch.nn as nn
from model.modules.modules import init_conv_random_normal, ScaleExp, init_conv_kaiming
from utill.utills import model_info
from model.backbone.resnet50 import ResNet50
# from model.od.proposed import HISFCOSHead
from model.backbone.efficientnetv1 import EfficientNetV1
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
    def __init__(self,
                 in_channel: List[int],
                 num_class: int,
                 feature: int,
                 freeze_bn: bool = True,
                 efficientnet: bool = False):
        super(FCOS, self).__init__()
        if efficientnet:
            self.backbone = EfficientNetV1(0)
        else:
            self.backbone = ResNet50(3)
        self.FPN = FeaturePyramidNetwork(in_channel, feature)
        # self.head = HeadFCOS(feature, num_class, 0.01)

        self.head = HeadFCOS(feature, num_class, 0.01)
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
            # self.backbone.freeze_stages(1)
            # print(f"success frozen_stage")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.FPN(x)
        cls, cnt, reg = self.head(x)
        return cls, cnt, reg


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channel: List[int], feature=256):
        super(FeaturePyramidNetwork, self).__init__()
        self.P5 = nn.Conv2d(in_channel[0], feature, 1, 1, 'same')
        self.P4 = nn.Conv2d(in_channel[1], feature, 1, 1, 'same')
        self.P3 = nn.Conv2d(in_channel[2], feature, 1, 1, 'same')
        self.P5_Up = nn.Upsample(scale_factor=2)
        self.P4_Up = nn.Upsample(scale_factor=2)
        self.P5_c1 = nn.Conv2d(feature, feature, 3, 1, 'same')
        self.P4_c1 = nn.Conv2d(feature, feature, 3, 1, 'same')
        self.P3_c1 = nn.Conv2d(feature, feature, 3, 1, 'same')
        self.P6_c1 = nn.Conv2d(feature, feature, 3, 2, 3//2)
        self.P7_c1 = nn.Conv2d(feature, feature, 3, 2, 3//2)
        self.act = nn.ReLU(True)
        self.apply(init_conv_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3, c4, c5 = x
        p5 = self.P5(c5)  # 16
        p4_c = self.P4(c4)  # 32
        p3_c = self.P3(c3)  # 64
        p4 = self.P5_Up(p5)  # 16 >32
        p4 = torch.add(p4, p4_c)   # 32+32
        p4 = self.P4_c1(p4)
        p3 = self.P4_Up(p4)
        p3 = torch.add(p3, p3_c)
        p3 = self.P3_c1(p3)
        p5 = self.P5_c1(p5)
        p6 = self.P6_c1(p5)
        p7 = self.P7_c1(self.act(p6))
        return p3, p4, p5, p6, p7


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

        self.cls_branch = nn.Sequential(*cls_branch)
        self.reg_branch = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(feature, self.class_num, kernel_size=3, padding=1)
        self.cnt_logits = nn.Conv2d(feature, 1, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(feature, 4, kernel_size=3, padding=1)

        self.apply(init_conv_random_normal)

        nn.init.constant_(self.cls_logits.bias, -np.log((1 - self.prior) / self.prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        for index, P in enumerate(inputs):
            cls_out = self.cls_branch(P)
            reg_out = self.reg_branch(P)
            cls_logits.append(self.cls_logits(cls_out))
            cnt_logits.append(self.cnt_logits(reg_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_out)))
        return cls_logits, cnt_logits, reg_preds


if __name__ == '__main__':  # flop51.69G -> 29M mAP ->  78.7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCOS(in_channel=[2048, 1024, 512], num_class=20, feature=256).to(device)
    model_info(model, 1, 3, 512, 512, device, depth=1)

    # from torch.utils.tensorboard import SummaryWriter
    # import os
    # writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', 'ef_fcos'))
    # writer.add_graph(model, tns)
    # writer.close()
