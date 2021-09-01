from model.backbone.resnet50 import ResNet50
import torch.nn as nn
import torch
from typing import List
from utills import model_info
from model.modules.modules import StdConv, DepthWiseConv2d, PointWiseConv


class Mc_FCOS(nn.Module):
    def __init__(self):
        super(Mc_FCOS, self).__init__()
        self.backbone = ResNet50(3)
        self.down_sample_1 = nn.MaxPool2d(2,2)
        self.ffm = FeatureFusionModule(feature_lv=[512,1024, 2048], feature = 256)
        self.tf1 = StdConv(2048, 256, 1, 1, 1//2)
        self.fpn = FPN(feautre = 256)
        # self.FFM = FeatureFusionModule(feature_lv = [512, 1024, 2048], feature = 256)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = self.backbone(x)  # 512, 1024, 2048
        x4 = self.down_sample_1(x3)
        x1, x2 = self.ffm([x1,x2,x3])
        x3 = self.tf1(x3)
        print(f'{x1.size()} {x2.size()} {x3.size()}')
        x1, x2, x3 = self.fpn([x1,x2,x3])
        return x1, x2, x3, x4


class MBConv(nn.Module):  # TODO SE Add
    def __init__(self,feature):
        super(MBConv, self).__init__()
        self.conv1 = PointWiseConv(in_channel = feature,
                                   out_channel = feature // 2)
        self.conv2 = DepthWiseConv2d(in_channel = feature // 2,
                                     kernel = 3)

        self.conv3 = PointWiseConv(in_channel = feature // 2,
                                   out_channel = feature // 2)

        self.conv4 = PointWiseConv(in_channel = feature // 2,
                                   out_channel = feature // 2)
        self.conv5 = PointWiseConv(in_channel = feature,
                                   out_channel = feature)
        self.bn = nn.BatchNorm2d(feature // 2)
        self.act = nn.SiLU(True)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv1(x)))
        x1 = x
        x = self.conv2(x)  # [1, 128, 16, 16])
        x = self.conv3(x)
        x1 = self.conv4(x1)
        x = torch.cat([x, x1], dim = 1)
        x = self.conv5(x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, feature_lv:List[int], feature:int):
        super(FeatureFusionModule, self).__init__()
        self.up1 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv1 = PointWiseConv(feature_lv[2] + feature_lv[1], feature)
        self.mb_conv1 = MBConv(feature)

        self.up2 = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv2 = PointWiseConv(feature_lv[1] + feature_lv[0], feature)
        self.mb_conv2 = MBConv(feature)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x
        x3 = self.up1(x3)

        x3 = torch.cat([x2,x3], dim = 1)
        x3 = self.conv1(x3)
        x3 = self.mb_conv1(x3)
        x2 = self.up2(x2)
        x2 = torch.cat([x1, x2], dim = 1)
        x2 = self.conv2(x2)
        x2 = self.mb_conv2(x2)
        return x2, x3


class FPN(nn.Module):
    def __init__(self, feautre:int):
        super(FPN, self).__init__()
        self.mb_conv1 = MBConv(feature = feautre)
        self.down_sample1 = nn.MaxPool2d(2, 2)
        self.mb_conv2 = MBConv(feature = feautre)
        self.down_sample2 = nn.MaxPool2d(2, 2)
        self.mb_conv3 = MBConv(feature = feautre)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x
        p1 = self.mb_conv1(x3)
        x3 = self.down_sample1(p1)
        x3 = torch.add(x2,x3)
        p2 = self.mb_conv2(x3)
        x3 = self.down_sample2(p2)
        x3 = torch.add(x1, x3)
        p3 = self.mb_conv3(x3)
        return p1,p2,p3


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # a = torch.randn(1, 256, 16, 16).to(device)
    # model = MBConv(256)
    # b = model(a)
    # print(f'{b.shape}')
    model = Mc_FCOS()
    print(model)
    model_info(model, 1, 3, 512, 512, device)


