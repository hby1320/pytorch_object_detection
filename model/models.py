import torch
import torchvision
import torch.onnx
import torch.nn as nn
from efficientnet_pytorch.utils import MemoryEfficientSwish
from efficientnet_pytorch import EfficientNet  # pip install efficientnet_pytorch
from torchinfo import summary
from model.backbone import resnet50
from model.modules.modules import *
from PIL import Image
from torchvision import transforms as t
import matplotlib.pyplot as plt


# def autopad(k:int, p=None) -> int:
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p

#  retina_net 비교군




class TestModel(nn.Module):
    def __init__(self, in_channel: int, backbone=''):
        super(TestModel, self).__init__()
        self.backbone = EfficientNet.from_pretrained(backbone, in_channels=in_channel, include_top=False)
        self.backbone.set_swish(memory_efficient=False)
        # self.Conv1 = depth_wise_Conv2d(1280,640,3)

        self.Conv1 = SeparableConv2d(1280, 640, 3)
        self.Conv2 = SeparableConv2d(640, 1280, 3)
        self.Conv3 = SeparableConv2d(1280, 640, 3)
        self.Conv4 = SeparableConv2d(640, 1280, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.extract_features(x)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        # x4 = fs['reduction_5']
        # x3 = fs['reduction_4']
        # x2 = fs['reduction_3']
        # x1 = fs['reduction_2']
        # x0 = fs['reduction_1']
        # x = self.Conv1(x4)
        # x = self.Conv1(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_list = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
                  'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    dummy_data = torch.randn(3, 512, 512)
