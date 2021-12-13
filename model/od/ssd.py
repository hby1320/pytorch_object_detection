import torch
import torch.nn as nn
from model.backbone.vgg16 import VGG16
from utill.utills import model_info


class SSD512(nn.Module):
    def __init__(self):
        super(SSD512, self).__init__()
        self.backbone = VGG16()
        self.block6 = nn.Conv2d(in_channels = 512,
                               out_channels = 1024,
                               kernel_size = 3,
                               stride = 1,
                               padding = 6,
                               dilation = 6)
        self.block7 = nn.Conv2d(in_channels = 1024,
                                out_channels = 1024,
                                kernel_size = 1,
                                stride = 1
                                )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        block4, block5 = self.backbone(x)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        return block4, block7


class SSDblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SSDblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_ch,
                               out_channels = out_ch//2,
                               kernel_size = 1,
                               stride = 1,
                               padding = 1)
        self.conv1 = nn.Conv2d(in_channels = out_ch,
                               out_channels = out_ch,
                               kernel_size = 1,
                               stride = 2,
                               padding = 1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSD512().to(device)
    dumy = torch.Tensor(1,3,512,512).to(device)
    model_info(model, 1,3,300,300,device)