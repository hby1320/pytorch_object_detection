import torch
import torch.nn as nn

from model.modules.modules import StdConv, PointWiseConv, DepthWiseConv2d
from model.backbone.resnet50 import ResNet50
from utill.utills import model_info
from typing import List


class CssFCOS(nn.Module):
    def __init__(self, feature_map:List[int], num_classes, feature):
        super(CssFCOS, self).__init__()
        self.backbone = ResNet50(re_layer = 3)
        self.down_sample_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Conv1 = StdConv(feature_map[2], feature, kernel = 3, st = 1,
                             padding = 3 // 2, actmode = 'swish')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1,x2,x3 = self.backbone(x)  # 512, 1024, 2048
        x3 = self.down_sample_1(x3)
        x3 = self.Conv1(x3)
        return [x1, x2, x3]


class FeaturePyramidNetwork(nn.Module):
    def __init__(self):
        super(FeaturePyramidNetwork, self).__init__()
        self.P6 = CssBlock()

class CssBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, beta:int =2, alpha:int =4):
        super(CssBlock, self).__init__()
        self.cross_stage = nn.Conv2d(in_channels = in_ch, out_channels = in_ch/2, kernel_size = 1, padding = 1,
                                     bias = False)
        self.point_wise_1 = PointWiseConv(in_channel = in_ch, out_channel = beta * in_ch)
        self.point_wise_2 = PointWiseConv(in_channel = in_ch / 2, out_channel = in_ch / 2)
        self.depth_wise_1 = DepthWiseConv2d(beta * in_ch, 3)
        self.se_block = SE_Block(2 * in_ch, alpha = alpha)



class SE_Block(nn.Module):
    def __init__(self, feature:int,  alpha:int = 4):
        super(SE_Block, self).__init__()
        self.Gap = nn.AdaptiveAvgPool2d
        self.point_wise_1 = PointWiseConv(in_channel = feature, out_channel = feature / alpha)
        self.point_wise_2 = PointWiseConv(in_channel = feature / alpha, out_channel = feature)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1 = self.Gap(x)
        x1 = x1.reshape(1,1,1)
        x1 = self.point_wise_1(x1)
        x1 = self.point_wise_2(x1)
        x1 = torch.mul(x,x1)
        x = torch.add(x,x1)
        return x







if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CssFCOS([512, 1024, 2048], 20, 512)
    model_info(model, 1, 3, 512, 512, device)



