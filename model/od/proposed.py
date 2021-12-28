import torch
import torch.nn as nn
from model.modules.modules import PointWiseConv, DepthWiseConv2d, ScaleExp, init_conv_random_normal, SEBlock
from model.backbone.resnet50 import ResNet50v2
from utill.utills import model_info
from typing import List
import numpy as np


class HalfInvertedStageFCOS(nn.Module):
    def __init__(self,
                 feature_map: List[int],
                 num_classes: int,
                 feature: int,
                 bn_freeze: bool = True):
        super(HalfInvertedStageFCOS, self).__init__()
        self.backbone = ResNet50v2()
        self.backbone_freeze = bn_freeze
        self.fpn = HalfInvertedStageFPN(feature_map, feature)
        self.head = HISFCOSHead(feature, num_classes, 0.01)

        def freeze_bn(module: nn.Module):
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
        x = self.backbone(x)  # 512 * 64, 1024 * 32, 2048  * 16
        x = self.fpn(x)  # p5, p6, p7 128 256 512
        cls, cnt, reg = self.head(x)
        return cls, cnt, reg


class HisBlock(nn.Module):
    def __init__(self, feature: int, beta: int = 4, d_rate: int = 2):
        super(HisBlock, self).__init__()
        self.conv1 = nn.Conv2d(feature, feature//2, 1, 1, 'same')
        self.conv2 = nn.Conv2d(feature, feature//2, 1, 1, 'same')
        self.conv3 = nn.Conv2d(feature, feature//2, 3, 1, 'same', bias=False)
        self.conv4 = nn.Conv2d(feature, feature, 3, 1, 'same', d_rate, bias=False)
        self.conv1_1 = DepthWiseConv2d(feature//2, 3, 1, False)
        self.conv1_2 = SEBlock(feature//2, beta)
        self.bn1 = nn.BatchNorm2d(feature//2)
        self.act1 = nn.SiLU(True)
        self.bn2 = nn.BatchNorm2d(feature//2)
        self.act2 = nn.ReLU(True)
        self.bn3 = nn.BatchNorm2d(feature//2)
        self.act3 = nn.ReLU(True)
        self.bn4 = nn.BatchNorm2d(feature)
        self.act4 = nn.SiLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x2 = self.conv2(x)
        x1_1 = self.conv1_1(x1)
        x1_1 = self.bn2(x1_1)
        x1_1 = self.act2(x1_1)
        x1_2 = self.conv1_2(x1)
        x1_c = torch.cat((x1_1, x1_2), dim=1)
        x1_c = self.conv3(x1_c)
        x1_c = self.bn3(x1_c)
        x1_c = self.act3(x1_c)
        x3 = torch.cat((x1_c, x2), dim=1)
        x3 = self.conv4(x3)
        x3 = self.bn4(x3)
        x3 = self.act4(x3)
        return x3


class HalfInvertedStageFPN(nn.Module):
    def __init__(self, feature_map: List[int], feature: int):
        super(HalfInvertedStageFPN, self).__init__()
        self.tf1 = nn.Conv2d(in_channels=feature_map[2], out_channels=feature, kernel_size=1, padding= 1//2, bias=False)
        self.tf2 = nn.Conv2d(feature_map[1], out_channels=feature, kernel_size=1, padding= 1//2, bias=False)
        self.tf3 = nn.Conv2d(feature_map[0], out_channels=feature, kernel_size=1, padding= 1//2, bias=False)
        self.HisBlock1 = HisBlock(feature, 4, 2)
        self.HisBlock2 = HisBlock(feature, 4, 2)
        self.HisBlock3 = HisBlock(feature, 4, 2)
        self.HisBlock4 = HisBlock(feature, 4, 2)
        self.HisBlock5 = HisBlock(feature, 4, 2)
        self.HisBlock6 = HisBlock(feature, 4, 2)
        self.HisBlock7 = HisBlock(feature, 4, 2)
        self.HisBlock8 = HisBlock(feature, 4, 2)
        self.HisBlock9 = HisBlock(feature, 4, 2)
        self.Up_sample1 = nn.Upsample(scale_factor=2)
        self.Up_sample2 = nn.Upsample(scale_factor=2)
        self.Up_sample3 = nn.Upsample(scale_factor=2)
        self.down_sample1 = nn.MaxPool2d(2, 2)
        self.down_sample2 = nn.MaxPool2d(2, 2)
        self.down_sample3 = nn.MaxPool2d(2, 2)
        self.down_sample4 = nn.MaxPool2d(2, 2)
        self.down_sample5 = nn.MaxPool2d(2, 2)
        self.down_sample6 = nn.MaxPool2d(2, 2)
        # self.gn1 = nn.GroupNorm(32, feature)
        # self.gn2 = nn.GroupNorm(32, feature)
        # self.gn3 = nn.GroupNorm(32, feature)
        self.gn1 = nn.BatchNorm2d(feature)
        self.gn2 = nn.BatchNorm2d(feature)
        self.gn3 = nn.BatchNorm2d(feature)
        self.act1 = nn.ReLU(True)
        self.act2 = nn.ReLU(True)
        self.act3 = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x1, x2, x3 = x
        # x3 = self.tf1(x3)  # 1024 16 16
        # p3 = self.icsp_blcok1(x3)  # 512 16 16  @
        # p3_1 = self.Up_sample1(p3)  # 512 32 32
        # x2 = self.tf2(x2)  # 512 32 32
        # p4_1 = torch.add(p3_1, x2)  # 512 32 32
        # p4 = self.icsp_blcok2(p4_1)  # 256 32 32   @
        # p5_1 = self.Up_sample2(p4)  # 256 64 64
        # x1 = self.tf3(x1)  # 256 64 64
        # p5_1 = torch.add(p5_1, x1)  # 256 64 64
        # p5 = self.icsp_blcok3(p5_1)  # 128 64 64 @
        # p5_2 = self.down_sample1(p5)
        # p4_2 = torch.add(p5_2, p4)
        # p4 = self.icsp_blcok4(p4_2)
        # p3_2 = self.down_sample2(p4)
        # p3 = torch.add(p3_2, p3)
        # p3 = self.icsp_blcok5(p3)
        # return p5, p4, p3
        # x1, x2, x3 = x
        # x3_1 = self.tf1(x3)  # 256 16 16
        # x3_1 = self.gn1(x3_1)  # 256 16 16
        # x3_1 = self.act1(x3_1)
        # x4_1 = self.down_sample3(x3_1)
        # x5_1 = self.down_sample4(x4_1)
        #
        # p3 = self.HisBlock1(x3_1)  # 256 16 16
        # p3_1 = self.Up_sample1(p3)  # 256 32 32
        # x2 = self.tf2(x2)  # 256 32 32
        # x2 = self.gn2(x2)
        # x2 = self.act2(x2)
        #
        # p4_1 = torch.add(p3_1, x2)  # 256 32 32
        #
        # p4 = self.HisBlock2(p4_1)  # 256 32 32
        # p5_1 = self.Up_sample2(p4)  # 256 64 64
        # x1 = self.tf3(x1)  # 256 64 64
        # x1 = self.gn2(x1)
        # x1 = self.act2(x1)
        #
        # p5_1 = torch.add(p5_1, x1)  # 256 64 64
        # p5 = self.HisBlock3(p5_1)  # 256 64 64 @
        #
        # p5_2 = self.down_sample1(p5)  # 32
        # p4_2 = torch.add(p5_2, p4)
        # p4 = self.HisBlock4(p4_2)  # 32
        #
        # p3_2 = self.down_sample2(p4)
        # p3 = torch.add(p3_2, p3)
        # p3 = self.HisBlock5(p3)  # 16
        #
        # p2_2 = self.down_sample5(p3)
        # p2 = torch.add(p2_2, x4_1)
        # p2 = self.HisBlock6(p2)
        #
        # p1_2 = self.down_sample6(p2)
        # p1 = torch.add(p1_2, x5_1)
        # p1 = self.HisBlock7(p1)
        # return p5, p4, p3, p2, p1
        x1, x2, x3 = x
        x3_1 = self.tf1(x3)  # 256 16 16
        x3_1 = self.gn1(x3_1)  # 256 16 16
        x3_1 = self.act1(x3_1)
        x4_1 = self.down_sample1(x3_1)  # 8
        x5_1 = self.down_sample2(x4_1)  # 4

        p3 = self.HisBlock1(x3_1)  # 256 16 16
        p3_1 = self.Up_sample1(p3)  # 256 32 32
        x2 = self.tf2(x2)  # 256 32 32
        x2 = self.gn2(x2)
        x2 = self.act2(x2)

        p4_1 = torch.add(p3_1, x2)  # 256 32 32

        p4 = self.HisBlock2(p4_1)  # 256 32 32
        p5_1 = self.Up_sample2(p4)  # 256 64 64
        x1 = self.tf3(x1)  # 256 64 64
        x1 = self.gn2(x1)
        x1 = self.act2(x1)

        p5_1 = torch.add(p5_1, x1)  # 256 64 64
        p5 = self.HisBlock3(p5_1)  # 256 64 64 @

        p5_2 = self.down_sample3(p5)  # 32
        p4_2 = torch.add(p5_2, p4)
        p4 = self.HisBlock4(p4_2)  # 32

        p3_2 = self.down_sample4(p4)
        p3 = torch.add(p3_2, p3)
        p3 = self.HisBlock5(p3)  # 16

        p2_2 = self.down_sample5(p3)
        p2 = torch.add(p2_2, x4_1)
        p2 = self.HisBlock6(p2)

        p1_2 = self.down_sample6(p2)
        p1 = torch.add(p1_2, x5_1)
        p1 = self.HisBlock7(p1)
        return p5, p4, p3, p2, p1


class HISFCOSHead(nn.Module):
    def __init__(self, feature: int, num_class: int, prior: float = 0.01):
        super(HISFCOSHead, self).__init__()
        self.class_num = num_class
        self.prior = prior
        cls_branch = []
        reg_branch = []
        self.pw1 = PointWiseConv(feature, 2*feature)
        self.pw2 = PointWiseConv(2 * feature, feature, bs=True)
        self.dw1 = DepthWiseConv2d(2*feature, 3)
        self.gn1 = nn.GroupNorm(32, 2 * feature)
        self.gn2 = nn.GroupNorm(32, 2 * feature)
        self.act1 = nn.ReLU(True)
        self.act2 = nn.SiLU(True)
        for i in range(1):
            cls_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding='same', dilation=3, bias=False))
            cls_branch.append(nn.GroupNorm(32, feature))
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(feature, feature, kernel_size=3, padding='same', bias=False))
            reg_branch.append(nn.GroupNorm(32, feature))
            reg_branch.append(nn.ReLU(True))

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(feature, self.class_num, kernel_size=3, padding=1)
        self.cnt_logits = nn.Conv2d(feature, 1, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(feature, 4, kernel_size=3, padding=1)
        self.apply(init_conv_random_normal)
        nn.init.constant_(self.cls_logits.bias, -np.log((1 - self.prior) / self.prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.2) for _ in range(5)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        for index, P in enumerate(inputs):
            x = self.pw1(P)
            x = self.gn1(x)
            x = self.act1(x)
            x = self.dw1(x)
            x = self.gn2(x)
            x = self.act2(x)
            x = self.pw2(x)
            P = torch.add(x, P)
            cls_conv_out = self.cls_conv(P)
            reg_conv_out = self.reg_conv(P)
            cls_logits.append(self.cls_logits(cls_conv_out))
            cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HalfInvertedStageFCOS([512, 1024, 2048], 20, 256).to(device)
    model_info(model, 1, 3, 512, 512, device)  # flop35.64G  para0.03G

    # from torch.utils.tensorboard import SummaryWriter
    # import os
    # writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', 'test_1'))
    # writer.add_graph(model, tns)
    # writer.close()
    """
      Total params: 32,662,846
      Trainable params: 32,378,366
      Non-trainable params: 284,480
      Total mult-adds (G): 36.91
      Input size (MB): 3.15
      Forward/backward pass size (MB): 1125.98
      Params size (MB): 130.65
      Estimated Total Size (MB): 1259.77
      """
