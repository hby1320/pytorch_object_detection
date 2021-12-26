import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from utill.utills import model_info
import numpy as np


class ResNet50(nn.Module):
    def __init__(self, re_layer=1):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained = True)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.max_pool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.re_layer = re_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.re_layer == 1:
            return x4
        elif self.re_layer == 2:
            return x3, x4
        elif self.re_layer == 3:
            return x2, x3, x4
        elif self.re_layer == 4:
            return x1, x2, x3, x4
        else:
            print(f' self.re_layer over range')
    """
    Input : 1, 3, 512, 512
    Total params: 23,508,032
    Trainable params: 23,508,032
    Non-trainable params: 0 
    Total mult-adds (G): 103.38
    """
    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self, 'layer{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


class ResNet50v2(nn.Module):
    def __init__(self):
        super(ResNet50v2, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained = True)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.layer1 = resnet50.layer1
        self.feature = ['layer2.3.relu_2', 'layer3.5.relu_2', 'layer4.2.relu_2']
        self.extract_feature = create_feature_extractor(resnet50, self.feature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_feature(x)
        return x[self.feature[0]], x[self.feature[1]], x[self.feature[2]]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = ResNet50(3).to(device)
    model = ResNet50v2()
    # model.freeze_bn()
    # model.freeze_stages(1)
    model_info(model, 1, 3, 512, 512, device)
    ResNet50v2()
