import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from utill.utills import model_info


class VGG16(nn.Module):  # SSD 300 / 512
    def __init__(self, pretrain: bool = True, ceil_mode: bool = True):
        super(VGG16, self).__init__()
        self.pretrain = pretrain
        self.ceil_mode = ceil_mode
        vgg16 = torchvision.models.vgg16(pretrained=self.pretrain)
        vgg16.features[16].ceil_mode = self.ceil_mode   #
        # vgg16.features[30].ceil_mode = True  #
        self.node = ['features.22', 'features.29']  # 'features.22': conv4-3, 'features.29': conv5-3 -> maxpool
        self.feature_extractor = create_feature_extractor(vgg16, return_nodes=self.node)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.act1 = nn.ReLU(True)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.act2 = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x[self.node[1]] = self.pool5(x[self.node[1]])
        x[self.node[1]] = self.conv6(x[self.node[1]])
        x[self.node[1]] = self.act1(x[self.node[1]])
        x[self.node[1]] = self.conv7(x[self.node[1]])
        x[self.node[1]] = self.act2(x[self.node[1]])
        return x[self.node[0]], x[self.node[1]]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16().to(device)
    model_info(model, 1, 3, 300, 300, device)