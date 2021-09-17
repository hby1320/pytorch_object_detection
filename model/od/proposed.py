import torch
import torch.nn as nn

from model.modules.modules import PointWiseConv, DepthWiseConv2d
from model.backbone.resnet50 import ResNet50
from utill.utills import model_info
from typing import List


class FRFCOS(nn.Module):
    def __init__(self, feature_map:List[int],
                 num_classes:int,
                 feature: int,
                 bn_freeze:bool = True):
        super(FRFCOS, self).__init__()
        self.backbone = ResNet50(re_layer = 3)
        self.backbone_freeze = bn_freeze
        self.conv1 = nn.Conv2d(in_channels = feature_map[2], out_channels = feature_map[0], kernel_size = 1, stride =1
                               ,bias = False)
        self.bn = nn.BatchNorm2d(feature_map[0])
        self.act = nn.SiLU(True)
        self.icsp_blcok1 = ICSPBlock(512, 512)
        self.icsp_blcok2 = ICSPBlock(512, 256)
        self.icsp_blcok3 = ICSPBlock(256, 128)
        self.Up_sample1 = nn.Upsample(scale_factor = 2)
        self.Up_sample2 = nn.Upsample(scale_factor = 2)
        self.refine = Refine_Module()
        self.head_s = RfFcosHead(128, feature, num_classes)
        self.head_m = RfFcosHead(256, feature, num_classes)
        self.head_l = RfFcosHead(512, feature, num_classes)

        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
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
        p5 = self.icsp_blcok1(x4)
        p6 = self.Up_sample1(p5)
        p6 = self.icsp_blcok2(p6)
        p7 = self.Up_sample2(p6)
        p7 = self.icsp_blcok3(p7)
        r3,r2,r1 = self.refine([p5,p6,p7])
        r1_cls,r1_cnt,r1_reg = self.head_s(r1)
        r2_cls,r2_cnt,r2_reg = self.head_m(r2)
        r3_cls,r3_cnt,r3_reg = self.head_l(r3)
        cls.append(r1_cls)
        cls.append(r2_cls)
        cls.append(r3_cls)
        cnt.append(r1_cnt)
        cnt.append(r2_cnt)
        cnt.append(r3_cnt)
        reg.append(r1_reg)
        reg.append(r2_reg)
        reg.append(r3_reg)

        return cls, cnt, reg


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
        # self.conv1 = nn.Conv2d(in_channels = in_ch, out_channels = in_ch, kernel_size = 3, padding = 1
        #                         , bias = False)
        self.se_block = SE_Block(in_ch, alpha = alpha)
        self.bn = nn.BatchNorm2d(beta * in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:  # [1, 512, 8, 8]
        x = self.bn(self.act(self.pw_conv1(x)))
        x = self.bn(self.act(self.dw_conv1(x)))
        x = self.pw_conv2(x)
        x1 = x
        x1 = self.se_block(x1)
        x1 = self.pw_conv3(x1)
        x = self.pw_conv5(x)
        x = torch.cat([x1,x],dim = 1)
        x = self.bn2(self.act(self.pw_conv4(x)))
        return x


class SE_Block(nn.Module):
    def __init__(self, feature:int,  alpha:int = 4):
        super(SE_Block, self).__init__()
        self.Gap = nn.AdaptiveAvgPool2d(1)
        self.pw_conv1 = PointWiseConv(in_channel = feature, out_channel = feature // 4, bs = True)
        self.pw_conv2 = PointWiseConv(in_channel = feature // 4, out_channel = feature, bs = True)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1 = self.Gap(x)  # torch.Size([1, 512, 1, 1])
        x1 = self.pw_conv1(x1)
        x1 = self.pw_conv2(x1)
        x1 = torch.mul(x,x1)
        x1 = torch.add(x1,x)
        return x1


class Refine_Module(nn.Module):
    def __init__(self):
        super(Refine_Module, self).__init__()
        self.up_sample1 = nn.Upsample(scale_factor = 2)
        self.up_sample2 = nn.Upsample(scale_factor = 2)
        self.down_sample1 = nn.MaxPool2d(2, 2)
        self.down_sample2 = nn.MaxPool2d(2, 2)
        self.pw_conv1 = PointWiseConv(768, 512, bs = True)
        self.pw_conv2 = PointWiseConv(896, 512, bs = True)
        self.pw_conv3 = PointWiseConv(512, 512, bs = True)
        self.pw_conv4 = PointWiseConv(512, 256, bs = True)
        self.pw_conv5 = PointWiseConv(512, 128, bs = True)
        self.icsp1 = ICSPBlock(512, 512)
        self.icsp2 = ICSPBlock(512, 512)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x  # hihg -> low ch

        # x1_up = self.up_sample1(x1)
        # x = torch.cat([x1_up,x2],dim = 1)
        #
        # x = self.pw_conv1(x)
        # x = self.icsp1(x)
        # x3_down = self.down_sample1(x3)
        # x = torch.cat([x,x3_down], dim = 1)
        # x = self.pw_conv2(x)
        # x = self.icsp2(x)
        x1_up = self.up_sample1(x1)
        # x = torch.cat([x1_up,x2],dim = 1)
        #
        # x = self.pw_conv1(x)
        # x = self.icsp1(x)
        x3_down = self.down_sample1(x3)
        x = torch.cat([x1_up, x, x3_down], dim = 1)
        x = self.pw_conv2(x)
        x = self.icsp2(x)

        ## Refine
        x_up = self.up_sample1(x)
        x_down = self.down_sample2(x)
        x_down = self.pw_conv3(x_down)
        x = self.pw_conv4(x)
        x_up = self.pw_conv5(x_up)

        x1 = torch.add(x_down, x1)
        x2 = torch.add(x, x2)
        x3 = torch.add(x_up, x3)
        return x1, x2, x3


class RfFcosHead(nn.Module):
    def __init__(self, head_lv, feature:int, num_class:int):
        super(RfFcosHead, self).__init__()
        self.reg_conv1 = nn.Conv2d(head_lv, feature, 3, 1, 1, bias = True)
        self.reg_conv2 = nn.Conv2d(feature, head_lv, 3, 1, 1, bias = True)
        self.cls_conv1 = nn.Conv2d(head_lv, feature, 3, 1, 1, bias = True)
        self.cls_conv2 = nn.Conv2d(feature, head_lv, 3, 1, 1, bias = True)
        self.reg = nn.Conv2d(head_lv, 4, 1, 1,bias = True)
        self.cls = nn.Conv2d(head_lv, num_class, 1, 1, bias = True)
        self.cnt = nn.Conv2d(head_lv, 1, 1, 1, bias = True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        reg_sub = self.reg_conv1(x)
        reg_sub = self.reg_conv2(reg_sub)

        cls_sub = self.reg_conv1(x)
        cls_sub = self.reg_conv2(cls_sub)
        cls = self.cls(cls_sub)
        cnt = self.cnt(reg_sub)
        reg = self.reg(reg_sub)
        return cls, cnt, reg

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FRFCOS([512, 1024, 2048], 20, 512)
    print(model)
    # a = torch.Tensor(1, 3, 512, 512)
    # b = model(a)

    model_info(model, 1, 3, 512, 512, device)
