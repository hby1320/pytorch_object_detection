import torch
import torch.nn as nn

from model.modules.modules import PointWiseConv, DepthWiseConv2d
from model.backbone.resnet50 import ResNet50
from utill.utills import model_info
from typing import List
import numpy as np


class FRFCOS(nn.Module):
    """
    Total params: 38,963,484
    Trainable params: 35,920,348
    Non-trainable params: 3,043,136
    Total mult-adds (G): 149.93
    Input size (MB): 3.15
    Forward/backward pass size (MB): 1198.24
    Params size (MB): 155.85
    Estimated Total Size (MB): 1357.24
    """
    def __init__(self, feature_map: List[int],
                 feature_lv: List[int],
                 num_classes: int,
                 feature: int,
                 bn_freeze: bool = True):
        super(FRFCOS, self).__init__()
        self.backbone = ResNet50(re_layer = 3)
        self.backbone_freeze = bn_freeze
        self.conv1 = nn.Conv2d(in_channels=feature_map[2], out_channels=feature_map[0], kernel_size=1,
                               stride=1, bias=False)
        self.bn = nn.BatchNorm2d(feature_map[0])
        self.act = nn.ReLU(True)
        self.fpn = ICSPFPN(feature_map, feature_lv)
        self.refine = RefineModule(feature_lv, 512)
        self.tf1 = PointWiseConv(feature_lv[2], feature)
        self.tf2 = PointWiseConv(feature_lv[1], feature)
        self.tf3 = PointWiseConv(feature_lv[0], feature)
        self.cls_net = ClassificationSub(feature, num_classes, 0.01)
        self.reg_net = RegressionSub(feature)
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(3)])

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
        cls = []
        cnt = []
        reg = []
        x1, x2, x3 = self.backbone(x)  # 512 * 64, 1024 * 32, 2048  * 16
        x4 = self.bn(self.act(self.conv1(x3)))  # 512 * 8
        x = self.fpn([x1,x2,x3,x4])
        x1, x2, x3 = self.refine(x)
        x1 = self.tf1(x1)
        x2 = self.tf2(x2)
        x3 = self.tf3(x3)
        x = [x1,x2,x3]
        for i, feature in enumerate(x):
            cls_logit = self.cls_net(feature)
            center_logit, reg_logit = self.reg_net(feature)
            cls.append(cls_logit)
            cnt.append(center_logit)
            reg.append(self.scale_exp[i](reg_logit))
        return cls, cnt, reg


class ICSPFPN(nn.Module):
    def __init__(self, feature_map: List[int], feature_lv: List[int]):
        super(ICSPFPN, self).__init__()
        self.tf1 = PointWiseConv(feature_map[2],feature_lv[2], bs= True)
        self.tf2 = PointWiseConv(feature_map[1],feature_lv[1], bs= True)
        self.tf3 = PointWiseConv(feature_map[0],feature_lv[0], bs= True)
        self.icsp_blcok1 = ICSPBlock(feature_lv[2], feature_lv[2])
        self.icsp_blcok2 = ICSPBlock(feature_lv[2], feature_lv[1])
        self.icsp_blcok3 = ICSPBlock(feature_lv[1], feature_lv[0])
        self.Up_sample1 = nn.Upsample(scale_factor = 2)
        self.Up_sample2 = nn.Upsample(scale_factor = 2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = x
        p5 = self.icsp_blcok1(x4)
        x3 = self.tf1(x3)
        p5 = torch.add(p5, x3)
        p6 = self.Up_sample1(p5)
        p6 = self.icsp_blcok2(p6)
        x2 = self.tf2(x2)
        p6 = torch.add(p6, x2)
        p7 = self.Up_sample2(p6)
        p7 = self.icsp_blcok3(p7)
        x1 = self.tf3(x1)
        p7 = torch.add(p7, x1)
        return p5, p6, p7


class ICSPBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, beta:int =2, alpha:int = 4):
        super(ICSPBlock, self).__init__()
        self.extention_ch = in_ch * beta
        self.pw_conv1 = PointWiseConv(in_channel = in_ch, out_channel = self.extention_ch)
        self.dw_conv1 = DepthWiseConv2d(self.extention_ch, 3, 1)
        self.pw_conv2 = PointWiseConv(in_channel = self.extention_ch, out_channel = in_ch)
        self.pw_conv3 = PointWiseConv(in_channel = in_ch, out_channel = in_ch)
        self.pw_conv4 = PointWiseConv(in_channel = in_ch * 2, out_channel = out_ch)
        self.pw_conv5 = PointWiseConv(in_channel = in_ch, out_channel = in_ch)
        self.se_block = SEBlock(in_ch, alpha = alpha)
        self.bn = nn.BatchNorm2d(self.extention_ch)
        self.bn1 = nn.BatchNorm2d(self.extention_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.bn4 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(True)
        self.act1 = nn.ReLU(True)
        self.act2 = nn.ReLU(True)
        self.act3 = nn.ReLU(True)
        self.act4 = nn.ReLU(True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:  # [1, 512, 8, 8]
        x = self.bn(self.act(self.pw_conv1(x)))
        x = self.bn1(self.act1(self.dw_conv1(x)))
        x = self.pw_conv2(x)
        x1 = x
        x1 = self.se_block(x1)
        x1 = self.bn2(self.act2(self.pw_conv3(x1)))
        x = self.bn3(self.act3(self.pw_conv5(x)))
        x = torch.cat([x1, x],dim = 1)
        x = self.bn4(self.act4(self.pw_conv4(x)))
        return x


class SEBlock(nn.Module):
    def __init__(self, feature:int,  alpha:int = 4):
        super(SEBlock, self).__init__()
        self.Gap = nn.AdaptiveAvgPool2d(1)
        self.pw_conv1 = PointWiseConv(in_channel = feature, out_channel = feature // 4, bs = True)
        self.act = nn.ReLU()
        self.pw_conv2 = PointWiseConv(in_channel = feature // 4, out_channel = feature, bs = True)
        self.sigmoid =nn.Sigmoid()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1 = self.Gap(x)  # torch.Size([1, 512, 1, 1])
        x1 = self.pw_conv1(x1)
        x1 = self.act(x1)
        x1 = self.pw_conv2(x1)
        x1 = self.sigmoid(x1)
        # x1 = torch.mul(x, x1)
        # x1 = torch.add(x1, x)
        return x1 * x


class RefineModule(nn.Module):
    def __init__(self, feature_lv: List[int], feature:int):
        super(RefineModule, self).__init__()
        self.up_sample1 = nn.Upsample(scale_factor = 2)
        self.up_sample2 = nn.Upsample(scale_factor = 2)
        self.down_sample1 = nn.MaxPool2d(2, 2)
        self.down_sample2 = nn.MaxPool2d(2, 2)
        self.pw_conv1 = PointWiseConv(feature_lv[2]+feature_lv[1], feature)
        self.pw_conv2 = PointWiseConv(feature_lv[2]+feature_lv[0], feature)
        self.pw_conv3 = PointWiseConv(feature, feature)
        self.pw_conv4 = PointWiseConv(feature, feature_lv[1])
        self.pw_conv5 = PointWiseConv(feature, feature_lv[0])
        self.icsp1 = ICSPBlock(feature, feature)
        self.icsp2 = ICSPBlock(feature, feature)
        self.act = nn.ReLU(True)
        self.act1 = nn.ReLU(True)
        self.act2 = nn.ReLU(True)
        self.act3 = nn.ReLU(True)
        self.act4 = nn.ReLU(True)
        self.bn = nn.BatchNorm2d(feature)
        self.bn1 = nn.BatchNorm2d(feature)
        self.bn2 = nn.BatchNorm2d(feature)
        self.bn3 = nn.BatchNorm2d(feature_lv[1])
        self.bn4 = nn.BatchNorm2d(feature_lv[0])

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x  # hihg -> low ch

        x1_up = self.up_sample1(x1)
        x = torch.cat([x1_up,x2],dim = 1)
        x = self.bn(self.act(self.pw_conv1(x)))
        x = self.icsp1(x)
        x3_down = self.down_sample1(x3)
        x = torch.cat([x,x3_down], dim = 1)
        x = self.bn1(self.act1(self.pw_conv2(x)))
        x = self.icsp2(x)
        ## Refine
        x_up = self.up_sample1(x)
        x_down = self.down_sample2(x)
        x_down = self.bn2(self.act2(self.pw_conv3(x_down)))
        x = self.bn3(self.act3(self.pw_conv4(x)))
        x_up = self.bn4(self.act4(self.pw_conv5(x_up)))
        x1 = torch.add(x_down, x1)
        x2 = torch.add(x, x2)
        x3 = torch.add(x_up, x3)

        return x1, x2, x3

#
# class RfFcosHead(nn.Module):
#     def __init__(self, head_lv, feature:int, num_class:int):
#         super(RfFcosHead, self).__init__()
#         self.reg_conv1 = nn.Conv2d(head_lv, feature, 3, 1, 1, bias = True)
#         self.reg_conv2 = nn.Conv2d(feature, head_lv, 3, 1, 1, bias = True)
#         self.cls_conv1 = nn.Conv2d(head_lv, feature, 3, 1, 1, bias = True)
#         self.cls_conv2 = nn.Conv2d(feature, head_lv, 3, 1, 1, bias = True)
#         self.reg = nn.Conv2d(head_lv, 4, 1, 1,bias = True)
#         self.cls = nn.Conv2d(head_lv, num_class, 1, 1, bias = True)
#         self.cnt = nn.Conv2d(head_lv, 1, 1, 1, bias = True)
#
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         reg_sub = self.reg_conv1(x)
#         reg_sub = self.reg_conv2(reg_sub)
#         cls_sub = self.cls_conv1(x)
#         cls_sub = self.cls_conv2(cls_sub)
#         cls = self.cls(cls_sub)
#         cnt = self.cnt(reg_sub)
#         reg = self.reg(reg_sub)
#         return cls, cnt, reg


class ClassificationSub(nn.Module):
    def __init__(self, feature, num_class, prior):
        super(ClassificationSub, self).__init__()
        self.prior = prior
        self.cls_c1 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1,bias=False)
        self.cls_c2 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1,bias=False)
        self.cls = nn.Conv2d(feature, num_class, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(feature)
        self.bn1 = nn.BatchNorm2d(feature)
        self.act = nn.ReLU(inplace = True)
        self.act1 = nn.ReLU(inplace = True)
        self.apply(init_conv_rand_nomal)
        nn.init.constant_(self.cls.bias, -np.log((1-self.prior)/self.prior))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.cls_c1(x)))
        x = self.act1(self.bn1(self.cls_c2(x)))
        cls = self.cls(x)
        return cls


class RegressionSub(nn.Module):
    def __init__(self,feature):
        super(RegressionSub, self).__init__()
        self.reg_c1 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
        self.reg_c2 = nn.Conv2d(feature, feature, kernel_size = 3, padding = 1, bias=False)
        self.reg = nn.Conv2d(feature, 4, kernel_size = 3, padding=1)
        self.center = nn.Conv2d(feature, 1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(feature)
        self.bn1 = nn.BatchNorm2d(feature)
        self.act = nn.ReLU(inplace = True)
        self.act1 = nn.ReLU(inplace=True)
        self.apply(init_conv_rand_nomal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.reg_c1(x)))
        x = self.act1(self.bn1(self.reg_c2(x)))
        reg = self.reg(x)
        center = self.center(x)
        return center, reg


class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value],dtype=torch.float32))

    def forward(self,x):
        return torch.exp(x * self.scale)


def init_conv_rand_nomal(module, std: int = 0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FRFCOS([512, 1024, 2048], [128, 256, 512], 20, 256).to(device)
    # tns = torch.rand(1,3, 512, 512).to(device)
    model_info(model, 1, 3, 512, 512, device)
    # from torch.utils.tensorboard import SummaryWriter
    # import os
    # writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', 'proposed'))
    #
    # writer.add_graph(model, tns)
    # writer.close()

