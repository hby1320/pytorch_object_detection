import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.vgg16 import VGG16
from utill.utills import model_info


class SSD512(nn.Module):
    """Implementation of the SSD VGG-based 512 network.
      The default features layers with 512x512 image input are:
        conv4 ==> 64 x 64
        conv7 ==> 32 x 32
        conv8 ==> 16 x 16
        conv9 ==> 8 x 8
        conv10 ==> 4 x 4
        conv11 ==> 2 x 2
        conv12 ==> 1 x 1
      The default image size used to train this network is 512x512.
      """
    def __init__(self):
        super(SSD512, self).__init__()
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)
        self.backbone = VGG16()
        self.maxpool = nn.MaxPool2d(kernel_size = 3,stride = 1, padding = 1, ceil_mode = False)
        self.block6 = nn.Conv2d(in_channels = 512,
                               out_channels = 1024,
                               kernel_size = 3,
                               stride = 1,
                               padding = 6,
                               dilation = 6)
        self.ReLU1 = nn.ReLU(inplace = True)
        self.block7 = nn.Conv2d(in_channels = 1024,
                                out_channels = 1024,
                                kernel_size = 1,
                                stride = 1
                                )
        self.ReLU2 = nn.ReLU(inplace = True)
        self.block8 = SSDBlock(1024, 512, 2)
        self.block9 = SSDBlock(512, 256, 2)
        self.block10 = SSDBlock2(256, 256)
        self.block11 = SSDBlock2(256, 256)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        block4, block5 = self.backbone(x)
        block4 = self.scale_weight.view(1, -1, 1, 1) * F.normalize(block4)
        block5 = self.maxpool(block5)
        block6 = self.block6(block5)
        block6 = self.ReLU1(block6)
        block7 = self.block7(block6)
        block7 = self.ReLU2(block7)
        block8 = self.block8(block7)
        block9 = self.block9(block8)
        block10 = self.block10(block9)
        block11 = self.block11(block10)

        return block4, block7, block8, block9, block10, block11


class SSDBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, st:int) -> torch.Tensor:
        super(SSDBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_ch,
                               out_channels = out_ch//2,
                               kernel_size = 1)
        self.ReLU1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_channels = out_ch//2,
                               out_channels = out_ch,
                               kernel_size = 3,
                               stride = st,
                               padding = 1)
        self.ReLU2 = nn.ReLU(inplace = True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.ReLU1(x)
        x = self.conv2(x)
        x = self.ReLU2(x)
        return x


class SSDBlock2(nn.Module):
    def __init__(self, in_ch:int, out_ch:int) -> torch.Tensor:
        super(SSDBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_ch,
                               out_channels = out_ch//2,
                               kernel_size = 1)
        self.ReLU1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_channels = out_ch//2,
                               out_channels = out_ch,
                               kernel_size = 3)
        self.ReLU2 = nn.ReLU(inplace = True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.ReLU1(x)
        x = self.conv2(x)
        x = self.ReLU2(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSD512().to(device)
    dumy = torch.Tensor(1,3,512,512).to(device)
    model_info(model, 1, 3, 300, 300,device)

    from torchvision.models.detection import ssd300_vgg16
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    #
    # anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    #                                        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
    #                                        steps=[8, 16, 32, 64, 100, 300])
    #
    # ssd = SSD(VGG16,anchor_generator, (512,512), 21)
    # ssd = ssd300_vgg16()
    # model_info(ssd, 1, 3, 300, 300, device, depth=5)