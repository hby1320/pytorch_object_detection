from model.backbone.resnet50 import ResNet50
import torch.nn as nn
import torch
from typing import List
from utill.utills import model_info
from model.modules.modules import StdConv, DepthWiseConv2d, PointWiseConv, SEBlock




class MC_FCOS(nn.Module):
    """
    # Total params: 34,418,880
    # Trainable params: 34,418,880
    # Non-trainable params: 0
    # Total mult-adds (G): 143.98
    # Input size (MB): 3.15
    # Forward/backward pass size (MB): 1117.88
    # Params size (MB): 137.68
    """
    def __init__(self,feature_lv:List[int], num_classes:int, feature:int):
        super(MC_FCOS, self).__init__()
        self.backbone = ResNet50(3)
        self.down_sample_1 = nn.MaxPool2d(2,2)
        self.mb_conv1 = MBConv(in_feature=2048,
                               out_feature=feature)
        self.ffm = FeatureFusionModule(feature_lv=feature_lv,
                                       features=feature)
        self.tf1 = StdConv(2048, feature, 1, 1, 1//2, 'swish')
        self.fpn = FPN(feautre=feature)
        self.refine = FeatureRefine(feature=feature)
        self.head = Detector_head(num_class = num_classes,
                                  feature=256)

        # self.FFM = FeatureFusionModule(feature_lv = [512, 1024, 2048], feature = 256)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = self.backbone(x)  # 512, 1024, 2048
        x4 = self.down_sample_1(x3)
        x4 = self.mb_conv1(x4)
        x1, x2 = self.ffm([x1,x2,x3])
        x3 = self.tf1(x3)
        x1, x2, x3 = self.fpn([x1,x2,x3])
        x = self.refine([x1, x2, x3, x4])
        cls = []
        cnt = []
        reg = []
        for i, feature in enumerate(x):
            cls_logits, cnt_logits, reg_logits = self.head(feature)
            cls.append(cls_logits)
            cnt.append(cnt_logits)
            reg.append(reg_logits)

        return cls, cnt, reg



class MBConv(nn.Module):  # TODO SE Add
    def __init__(self, in_feature:int, out_feature:int, r=6):
        super(MBConv, self).__init__()
        # expanded = expansion_factor * in_feature
        self.conv1 = PointWiseConv(in_channel = in_feature,
                                   out_channel = in_feature // 2)
        self.conv2 = DepthWiseConv2d(in_channel = in_feature // 2,
                                     kernel = 3)
        self.se = SEBlock(in_feature // 2, r = r)
        self.conv3 = PointWiseConv(in_channel = in_feature // 2,
                                   out_channel = in_feature // 2)
        # self.se =
        self.conv4 = PointWiseConv(in_channel = in_feature // 2,
                                   out_channel = in_feature // 2)
        self.conv5 = PointWiseConv(in_channel = in_feature,
                                   out_channel = out_feature)

        self.bn = nn.BatchNorm2d(in_feature // 2)
        self.bn2 = nn.BatchNorm2d(out_feature)
        self.act = nn.SiLU(True)  #swich
        # self.Sigmoid = nn.Sigmoid


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv1(x))) ## channel sq  # CSP  c/2
        x1 = x
        x = self.act(self.bn(self.conv2(x)))  # [1, 128, 16, 16]) ## Dw
        x = self.se(x)
        x = self.act(self.bn(self.conv3(x)))
        x1 = self.act(self.bn(self.conv4(x1)))
        x = torch.cat([x, x1], dim = 1)
        x = self.act(self.bn2(self.conv5(x)))
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, feature_lv:List[int], features:int):
        super(FeatureFusionModule, self).__init__()
        self.up1 = nn.Upsample(scale_factor = 2,
                               mode = 'nearest')
        self.conv1 = PointWiseConv(in_channel = feature_lv[2] + feature_lv[1],
                                   out_channel = features)
        self.mb_conv1 = MBConv(in_feature = features,
                               out_feature = features)

        self.up2 = nn.Upsample(scale_factor = 2,
                               mode = 'nearest')
        self.conv2 = PointWiseConv(in_channel = feature_lv[1] + feature_lv[0],
                                   out_channel = features)
        self.mb_conv2 = MBConv(in_feature = features,
                               out_feature = features)

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
        self.mb_conv1 = MBConv(in_feature = feautre,
                               out_feature = feautre)
        self.up_sample1 = nn.Upsample(scale_factor = 2)
        self.mb_conv2 = MBConv(in_feature = feautre,
                               out_feature = feautre)
        self.up_sample2 = nn.Upsample(scale_factor = 2)
        self.mb_conv3 = MBConv(in_feature = feautre,
                               out_feature = feautre)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = x
        p1 = self.mb_conv1(x3)
        x3 = self.up_sample1(p1)
        x3 = torch.add(x2,x3)
        p2 = self.mb_conv2(x3)
        x3 = self.up_sample2(p2)
        x3 = torch.add(x1, x3)
        p3 = self.mb_conv3(x3)
        return p1, p2, p3


class FeatureRefine(nn.Module):
    def __init__(self, feature=256):
        super(FeatureRefine, self).__init__()
        self.mb_conv1 = MBConv(feature*2,
                              feature)
        self.up_sample1 = nn.Upsample(scale_factor = 2)
        self.mb_conv2 = MBConv(feature*2,
                              feature)
        self.down_sample1 = nn.MaxPool2d(kernel_size = 2,
                                         stride = 2)
        self.down_sample2 = nn.MaxPool2d(kernel_size = 2,
                                         stride = 2)
        self.down_sample3 = nn.MaxPool2d(kernel_size = 2,
                                         stride = 2)
        self.up_sample2 = nn.Upsample(scale_factor = 2)

        self.conv1 = nn.Conv2d(in_channels = feature,
                               out_channels = feature,
                               kernel_size = 3,
                               padding = 1,
                               stride = 1,
                               bias = False)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = x  #  [1, 256, 16, 16] [1, 256, 32, 32] [1, 256, 64, 64] [1, 2048, 8, 8]
        c1 = self.up_sample1(x1)
        c1 = torch.cat([c1,x2], dim = 1)
        c1 = self.mb_conv1(c1)
        c2 = self.down_sample1(x3)
        c1 = torch.cat([c1,c2], dim = 1)
        c1 = self.mb_conv2(c1)
        ## feature refine
        h1 = self.up_sample2(c1)  #  [1, 256, 64, 64]
        h1 = torch.add(h1, x3)
        h2 = self.conv1(c1)  # [1, 256, 32, 32]
        h2 = torch.add(h2, x2)
        h3_1 = self.down_sample2(c1)  # [1, 256, 16, 16]
        h3 = torch.add(h3_1, x1)
        h4 = self.down_sample3(h3_1)  # [1, 256, 8, 8]
        h4 = torch.add(h4, x4)

        return h1, h2, h3, h4


class Detector_head(nn.Module):
    def __init__(self,num_class:int,feature =256 ):
        super(Detector_head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = feature, out_channels = feature,
                               kernel_size = 3,padding = 3//2, padding_mode = 'zeros',
                               bias = False)
        self.conv2 = nn.Conv2d(in_channels = feature, out_channels = feature,
                               kernel_size = 3, padding = 3//2, padding_mode = 'zeros',
                               bias = False)
        self.conv3 = nn.Conv2d(in_channels = feature, out_channels = feature,
                               kernel_size = 3, padding = 3//2, padding_mode = 'zeros',
                               bias = False)
        self.conv4 = nn.Conv2d(in_channels = feature, out_channels = feature,
                               kernel_size = 3, padding = 3//2, padding_mode = 'zeros',
                               bias = False)
        self.cls = nn.Conv2d(in_channels = feature, out_channels = num_class,
                             kernel_size = 3, padding = 3//2, padding_mode = 'zeros',
                             bias = False)
        self.cnt = nn.Conv2d(in_channels = feature, out_channels = 1,
                             kernel_size = 3, padding = 3//2, padding_mode = 'zeros',
                             bias = False)
        self.reg = nn.Conv2d(in_channels = feature, out_channels = 4,
                             kernel_size = 3, padding = 3//2, padding_mode = 'zeros',
                             bias = False)
        self.bn = nn.BatchNorm2d(feature)
        self.silu = nn.SiLU(True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.silu(self.bn(self.conv1(x)))
        x = self.silu(self.bn(self.conv2(x)))
        x = self.silu(self.bn(self.conv3(x)))
        x = self.silu(self.bn(self.conv4(x)))
        cls = self.cls(x)
        cnt = self.cnt(x)
        reg = self.reg(x)
        return cls, cnt, reg


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MC_FCOS([512, 1024, 2048], 20, 256)
    model_info(model, 1, 3, 512, 512, device)
    z = torch.rand(1,3,512,512).to(device)

    a,b,c = model(z)


