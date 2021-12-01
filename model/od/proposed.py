import torch
import torch.nn as nn
from model.modules.modules import PointWiseConv, DepthWiseConv2d
from model.backbone.resnet50 import ResNet50
from model.backbone.efficientnetv1 import EfficientNetV1
from utill.utills import model_info
from typing import List
import numpy as np


class FRFCOS(nn.Module):
    """
Total params: 38,465,372
Trainable params: 38,154,524
Non-trainable params: 310,848
Total mult-adds (G): 49.14
Input size (MB): 3.15
Forward/backward pass size (MB): 1376.77
Params size (MB): 153.86
Estimated Total Size (MB): 1533.78

    flop44.98G  para0.03G
Total GFLOPs: 98.2822
    """

    def __init__(self, feature_map: List[int],
                 num_classes: int,
                 feature: int,
                 bn_freeze: bool = True):
        super(FRFCOS, self).__init__()
        self.backbone = ResNet50(3)
        # self.backbone = EfficientNetV1(1)
        self.backbone_freeze = bn_freeze
        self.fpn = ICSPFPN(feature_map, feature)
        # self.ifpn = IFPN(feature_map, feature)
        self.head = HeadFRFCOS(feature, num_classes, 0.01)

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
        # x = self.ifpn(x)
        cls, cnt, reg = self.head(x)
        return cls, cnt, reg


class IFPN(nn.Module):
    def __init__(self, feature_map: List[int], feature: int):
        super(IFPN, self).__init__()
        self.tf1 = nn.Conv2d(in_channels=feature_map[2], out_channels=feature, kernel_size=1, padding= 1//2, bias=False)
        self.tf2 = nn.Conv2d(feature_map[1], out_channels=feature, kernel_size=1, padding= 1//2, bias=False)
        self.tf3 = nn.Conv2d(feature_map[0], out_channels=feature, kernel_size=1, padding= 1//2, bias=False)
        self.inverted_blcok1 = MCbottle(feature, feature)
        self.inverted_blcok2 = MCbottle(feature, feature)
        self.inverted_blcok3 = MCbottle(feature, feature)
        self.inverted_blcok4 = MCbottle(feature, feature)
        self.inverted_blcok5 = MCbottle(feature, feature)
        self.inverted_blcok6 = MCbottle(feature, feature)
        self.inverted_blcok7 = MCbottle(feature, feature)
        self.inverted_blcok8 = MCbottle(feature, feature)
        self.inverted_blcok9 = MCbottle(feature, feature)
        self.inverted_blcok10 = MCbottle(feature, feature)
        self.Up_sample1 = nn.Upsample(scale_factor=2)
        self.Up_sample2 = nn.Upsample(scale_factor=2)
        self.down_sample1 = nn.MaxPool2d(2, 2)
        self.down_sample2 = nn.MaxPool2d(2, 2)
        # self.down_sample3 = nn.MaxPool2d(2, 2)
        # self.conv1 = nn.Conv2d(feature,feature, 3,1,1//2,bias=False)
        # self.bn = nn.BatchNorm2d(feature)
        # # self.bn1 = nn.BatchNorm2d(feature)
        # # self.bn2 = nn.BatchNorm2d(feature)
        # self.act = nn.ReLU(True)
        # self.act1 = nn.ReLU(True)
        # self.act2 = nn.ReLU(True)

    # [512, 1024, 2048], [128, 256, 512]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x
        x3 = self.tf1(x3)  # 1024 16 16
        p3 = self.inverted_blcok1(x3)  # 512 16 16  @
        p3 = self.inverted_blcok2(p3)
        p3_1 = self.Up_sample1(p3)  # 512 32 32
        x2 = self.tf2(x2)  # 512 32 32
        p4_1 = torch.add(p3_1, x2)  # 512 32 32
        p4 = self.inverted_blcok3(p4_1)  # 256 32 32   @
        p4 = self.inverted_blcok4(p4)
        p5_1 = self.Up_sample2(p4)  # 256 64 64
        x1 = self.tf3(x1)  # 256 64 64
        p5_1 = torch.add(p5_1, x1)  # 256 64 64
        p5 = self.inverted_blcok5(p5_1)  # 128 64 64 @
        p5 = self.inverted_blcok6(p5)
        p5_2 = self.down_sample1(p5)
        p4_2 = torch.add(p5_2, p4)
        p4 = self.inverted_blcok7(p4_2)
        p4 = self.inverted_blcok8(p4)
        p3_2 = self.down_sample2(p4)
        p3 = torch.add(p3_2, p3)
        p3 = self.inverted_blcok9(p3)
        p3 = self.inverted_blcok10(p3)
        return p5, p4, p3


# class ICSPFPN(nn.Module):
#     def __init__(self, feature_map: List[int], feature: int):
#         super(ICSPFPN, self).__init__()
#         self.tf1 = nn.Conv2d(in_channels=feature_map[2], out_channels=feature, kernel_size=1, padding= 1//2, bias=True)
#         self.tf2 = nn.Conv2d(feature_map[1], out_channels=feature, kernel_size=1, padding= 1//2, bias=True)
#         self.tf3 = nn.Conv2d(feature_map[0], out_channels=feature, kernel_size=1, padding= 1//2, bias=True)
#         self.icsp_blcok1 = ICSPBlock(feature, feature, 3, 2, 4)
#         self.icsp_blcok2 = ICSPBlock(feature, feature, 3, 2, 4)
#         self.icsp_blcok3 = ICSPBlock(feature, feature, 3, 2, 4)
#         self.icsp_blcok4 = ICSPBlock(feature, feature, 3, 2, 4)
#         self.icsp_blcok5 = ICSPBlock(feature, feature, 3, 2, 4)
#         self.Up_sample1 = nn.Upsample(scale_factor=2)
#         self.Up_sample2 = nn.Upsample(scale_factor=2)
#
#         # self.down_sample1 = nn.Conv2d(feature,feature, 3 , 2, 3//2)
#         # self.down_sample2 = nn.Conv2d(feature,feature, 3 , 2, 3//2)
#         self.down_sample1 = nn.MaxPool2d(2, 2)
#         self.down_sample2 = nn.MaxPool2d(2, 2)
#         # self.down_sample3 = nn.MaxPool2d(2, 2)
#         # self.conv1 = nn.Conv2d(feature,feature, 3,1,1//2,bias=False)
#         # self.bn = nn.BatchNorm2d(feature)
#         # # self.bn1 = nn.BatchNorm2d(feature)
#         # # self.bn2 = nn.BatchNorm2d(feature)
#         # self.act = nn.ReLU(True)
#         # self.act1 = nn.ReLU(True)
#         # self.act2 = nn.ReLU(True)
#
#     # [512, 1024, 2048], [128, 256, 512]
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x1, x2, x3 = x
#         x3 = self.tf1(x3)  # 1024 16 16
#
#         p3 = self.icsp_blcok1(x3)  # 512 16 16  @
#         p3_1 = self.Up_sample1(p3)  # 512 32 32
#         x2 = self.tf2(x2)  # 512 32 32
#         p4_1 = torch.add(p3_1, x2)  # 512 32 32
#         p4 = self.icsp_blcok2(p4_1)  # 256 32 32   @
#         p5_1 = self.Up_sample2(p4)  # 256 64 64
#         x1 = self.tf3(x1)  # 256 64 64
#         p5_1 = torch.add(p5_1, x1)  # 256 64 64
#         p5 = self.icsp_blcok3(p5_1)  # 128 64 64 @
#
#         p5_2 = self.down_sample1(p5)
#         p4_2 = torch.add(p5_2, p4)
#         p4 = self.icsp_blcok4(p4_2)
#         p3_2 = self.down_sample2(p4)
#         p3 = torch.add(p3_2, p3)
#         p3 = self.icsp_blcok5(p3)
#         return p5, p4, p3


class TestModule(nn.Module):
    def __init__(self, feature: int, beta: int = 4):
        super(TestModule, self).__init__()
        self.conv1 = nn.Conv2d(feature, feature//2, 1, 1, 1//2)
        self.conv2 = nn.Conv2d(feature, feature//2, 1, 1, 1//2)
        self.conv3 = nn.Conv2d(feature, feature//2, 3, 1, 3//2)
        self.conv4 = nn.Conv2d(feature, feature, 3, 1, 1, bias=False)
        self.conv1_1 = DepthWiseConv2d(feature//2, 3, 1)
        self.conv1_2 = SEBlock(feature//2, alpha=beta)
        self.bn1 = nn.BatchNorm2d(feature//2)
        self.act1 = nn.SiLU(True)
        self.bn2 = nn.BatchNorm2d(feature//2)
        self.act2 = nn.ReLU(True)
        self.bn3 = nn.BatchNorm2d(feature//2)
        self.act3 = nn.SiLU(True)
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
        x1_c = torch.add(x1_c, x)
        x1_c = self.conv3(x1_c)
        x1_c = self.bn3(x1_c)
        x1_c = self.act3(x1_c)
        x3 = torch.cat((x1_c, x2), dim=1)
        x3 = self.conv4(x3)
        x3 = self.bn4(x3)
        x3 = self.act4(x3)
        return x3


class ICSPFPN(nn.Module):
    def __init__(self, feature_map: List[int], feature: int):
        super(ICSPFPN, self).__init__()
        self.tf1 = nn.Conv2d(in_channels=feature_map[2], out_channels=feature, kernel_size=1, padding= 1//2, bias=True)
        self.tf2 = nn.Conv2d(feature_map[1], out_channels=feature, kernel_size=1, padding= 1//2, bias=True)
        self.tf3 = nn.Conv2d(feature_map[0], out_channels=feature, kernel_size=1, padding= 1//2, bias=True)

        # self.icsp_blcok1 = ICSPBlock(feature, feature, 3, 2, 4)
        # self.icsp_blcok2 = ICSPBlock(feature, feature, 3, 2, 4)
        # self.icsp_blcok3 = ICSPBlock(feature, feature, 3, 2, 4)
        # self.icsp_blcok4 = ICSPBlock(feature, feature, 3, 2, 4)
        # self.icsp_blcok5 = ICSPBlock(feature, feature, 3, 2, 4)
        # self.icsp_blcok6 = ICSPBlock(feature, feature, 3, 2, 4)
        # self.icsp_blcok7 = ICSPBlock(feature, feature, 3, 2, 4)
        self.icsp_blcok1 = TestModule(feature, 4)
        self.icsp_blcok2 = TestModule(feature, 4)
        self.icsp_blcok3 = TestModule(feature, 4)
        self.icsp_blcok4 = TestModule(feature, 4)
        self.icsp_blcok5 = TestModule(feature, 4)
        self.icsp_blcok6 = TestModule(feature, 4)
        self.icsp_blcok7 = TestModule(feature, 4)
        self.Up_sample1 = nn.Upsample(scale_factor=2)
        self.Up_sample2 = nn.Upsample(scale_factor=2)
        self.Up_sample3 = nn.Upsample(scale_factor=2)
        # self.down_sample1 = nn.Conv2d(feature,feature, 3 , 2, 3//2)
        # self.down_sample2 = nn.Conv2d(feature,feature, 3 , 2, 3//2)
        self.down_sample1 = nn.MaxPool2d(2, 2)
        self.down_sample2 = nn.MaxPool2d(2, 2)
        self.down_sample3 = nn.MaxPool2d(2, 2)
        # self.down_sample3 = nn.MaxPool2d(2, 2)
        # self.conv1 = nn.Conv2d(feature,feature, 3,1,1//2,bias=False)
        # self.bn = nn.BatchNorm2d(feature)
        # # self.bn1 = nn.BatchNorm2d(feature)
        # # self.bn2 = nn.BatchNorm2d(feature)
        # self.act = nn.ReLU(True)
        # self.act1 = nn.ReLU(True)
        # self.act2 = nn.ReLU(True)

    # [512, 1024, 2048], [128, 256, 512]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x0, x1, x2, x3 = x
        x1, x2, x3 = x
        x3 = self.tf1(x3)  # 1024 16 16
        p3 = self.icsp_blcok1(x3)  # 512 16 16  @
        p3_1 = self.Up_sample1(p3)  # 512 32 32
        x2 = self.tf2(x2)  # 512 32 32
        p4_1 = torch.add(p3_1, x2)  # 512 32 32
        p4 = self.icsp_blcok2(p4_1)  # 256 32 32   @
        p5_1 = self.Up_sample2(p4)  # 256 64 64
        x1 = self.tf3(x1)  # 256 64 64
        p5_1 = torch.add(p5_1, x1)  # 256 64 64
        p5 = self.icsp_blcok3(p5_1)  # 128 64 64 @
        # # head test #
        # p6_1 = self.Up_sample3(p5)
        # p6_1 = torch.add(p6_1, x0)
        # p6 = self.icsp_blcok4(p6_1)
        # p5_2 = self.down_sample1(p6)
        # p5_2 = torch.add(p5_2, x1)
        # p5 = self.icsp_blcok5(p5_2)
        # p4_2 = self.down_sample2(p5)
        # p4_2 = torch.add(p4_2, x2)
        # p4 = self.icsp_blcok6(p4_2)
        # p3_2 = self.down_sample3(p4)
        # p3_2 = torch.add(p3_2, x3)
        # p3 = self.icsp_blcok6(p3_2)
        p5_2 = self.down_sample1(p5)
        p4_2 = torch.add(p5_2, p4)
        p4 = self.icsp_blcok4(p4_2)
        p3_2 = self.down_sample2(p4)
        p3 = torch.add(p3_2, p3)
        p3 = self.icsp_blcok5(p3)

        # return p6, p5, p4, p3
        return p5, p4, p3


class MCbottle(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, beta: int = 4, alpha: int = 4):
        super(MCbottle, self).__init__()
        self.conv1_1 = PointWiseConv(in_channel=in_ch, out_channel=in_ch * beta)
        self.conv1_2 = DepthWiseConv2d(in_ch * beta, k, 1)
        self.conv1_3 = PointWiseConv(in_channel=in_ch * beta, out_channel=out_ch)
        self.se_block = SEBlock(in_ch * beta, alpha=alpha)
        self.bn = nn.BatchNorm2d(in_ch * beta)
        self.bn1 = nn.BatchNorm2d(in_ch * beta)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(True)
        self.act1 = nn.SiLU(True)
        self.act2 = nn.SiLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.bn(self.conv1_1(x)))
        x1 = self.act1(self.bn1(self.conv1_2(x1)))
        x1 = self.se_block(x1)
        x1 = self.act2(self.bn2(self.conv1_3(x1)))
        x1 = torch.add(x1, x)
        return x1


class ICSPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, beta: int = 2, alpha: int = 4):
        super(ICSPBlock, self).__init__()
        # self.pw_conv1 = PointWiseConv(in_channel=in_ch, out_channel=in_ch*beta)
        # self.pw_conv2 = PointWiseConv(in_channel=in_ch*beta, out_channel=in_ch)
        self.bottle_1 = MCbottle(in_ch, in_ch, k, beta, alpha)
        self.bottle_2 = MCbottle(in_ch, in_ch, k, beta, alpha)
        # self.bottle_3 = MCbottle(in_ch, in_ch, k, beta, alpha)
        self.pw_conv3 = PointWiseConv(in_channel=in_ch, out_channel=in_ch // 2)
        self.pw_conv4 = PointWiseConv(in_channel=in_ch, out_channel=in_ch // 2)
        self.pw_conv5 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, 1, bias=False)
        # self.dw_conv1 = DepthWiseConv2d(in_ch*beta, k, 1)
        # self.se_block = SEBlock(in_ch*beta, alpha=alpha)
        # self.bn = nn.BatchNorm2d(in_ch*beta)
        # self.bn1 = nn.BatchNorm2d(in_ch*beta)
        # self.bn2 = nn.BatchNorm2d(in_ch)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.bn4 = nn.BatchNorm2d(out_ch)
        # self.act = nn.ReLU(True)
        # self.act1 = nn.ReLU(True)
        # self.act2 = nn.ReLU(True)
        self.act3 = nn.ReLU(True)
        self.act4 = nn.ReLU(True)
        # self.conv1 = nn.Conv2d(in_ch, in_ch*beta, 3, 1,1,bias=False)
        # self.bn5 = nn.BatchNorm2d(in_ch*beta)
        # self.act5 = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [1, 512, 8, 8]
        x1 = self.bottle_1(x)
        x1 = self.bottle_2(x1)
        x2 = self.pw_conv3(x1)
        x3 = self.pw_conv4(x)
        x3 = self.act3(self.bn3(torch.cat([x2, x3], dim=1)))
        x3 = self.act4(self.bn4(self.pw_conv5(x3)))
        return x3


class SEBlock(nn.Module):
    def __init__(self, feature: int, alpha: int = 4):
        super(SEBlock, self).__init__()
        self.Gap = nn.AdaptiveAvgPool2d(1)
        self.pw_conv1 = PointWiseConv(in_channel=feature, out_channel=feature // alpha, bs=True)
        self.act = nn.SiLU(True)
        self.pw_conv2 = PointWiseConv(in_channel=feature // alpha, out_channel=feature, bs=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.Gap(x)  # torch.Size([1, 512, 1, 1])
        x1 = self.pw_conv1(x1)
        x1 = self.act(x1)
        x1 = self.pw_conv2(x1)
        x1 = self.sigmoid(x1)
        return x1 * x


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)


def init_conv_randomnormal(module: nn.Module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class HeadFRFCOS(nn.Module):
    def __init__(self, feature: int, num_class: int, prior: float = 0.01):
        super(HeadFRFCOS, self).__init__()
        self.class_num = num_class
        self.prior = prior
        cls_branch = []
        reg_branch = []
        self.pw1 = PointWiseConv(feature, 2*feature)
        self.pw2 = PointWiseConv(2 * feature, feature ,bs=True)
        self.dw1 = DepthWiseConv2d(2*feature, 3)
        self.gn1 = nn.GroupNorm(32, 2 * feature)
        self.gn2 = nn.GroupNorm(32, 2 * feature)
        self.act1 = nn.ReLU(True)
        self.act2 = nn.SiLU(True)

        for i in range(1):
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
        self.apply(init_conv_randomnormal)
        nn.init.constant_(self.cls_logits.bias, -np.log((1 - self.prior) / self.prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(4)])

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Total params: 34,136,668  flop39.39G  para0.03G
    model = FRFCOS([512, 1024, 2048], 20, 256).to(device)
    # model = FRFCOS([40, 112, 320], 20, 256).to(device)
    model_info(model, 1, 3, 512, 512, device)  # flop35.64G  para0.03G
    # tns = torch.rand(1, 3, 512, 512).to(device)
    #
    # from torch.utils.tensorboard import SummaryWriter
    # import os
    # writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', 'test_1'))
    # # #
    # writer.add_graph(model, tns)
    # writer.close()
    # Total
    # params: 38, 465, 372
    # Trainable
    # params: 38, 154, 524
    # Non - trainable
    # params: 310, 848
    # Total
    # mult - adds(G): 49.14