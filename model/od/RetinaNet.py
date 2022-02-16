import torch
import torch.nn as nn
from model.modules.modules import StdConv
from model.backbone.resnet50 import ResNet50
from utill.utills import model_info, shift_xy, generate_anchor
import numpy as np


class RetinaNet(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        self.backbone = ResNet50(re_layer = 3)
        self.fpn = FeaturePyramid(512, 1024, 2048, feature_size=256)
        self.regression_sub_net = RegressionSubNet(256)
        self.classification_sub_net = ClassificationSubNet(256, num_class=num_class)
        self.anchor = Anchor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fpn(x)
        regression = torch.cat([self.regression_sub_net(feature) for feature in x], dim=1)
        classification = torch.cat([self.classification_sub_net(feature) for feature in x], dim=1)
        # anchors = self.anchor(x)
        # return regression,classification,anchors
        return regression, classification


class FeaturePyramid(nn.Module):
    def __init__(self, c3_size: int, c4_size: int, c5_size: int, feature_size=256):
        super(FeaturePyramid, self).__init__()
        self.P5_1 = StdConv(c5_size, feature_size, 1, 1, 0)
        self.P5_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = StdConv(feature_size, feature_size, 3, 1, 1)
        self.P4_1 = StdConv(c4_size, feature_size, 1, 1, 0)
        self.P4_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = StdConv(feature_size, feature_size, 3, 1, 1)
        self.P3_1 = StdConv(c3_size, feature_size, 1, 1, 0)
        self.P3_2 = StdConv(feature_size, feature_size, 3, 1, 1)
        self.P6 = StdConv(c5_size, feature_size, 3, 2, 1)
        self.P7_1 = nn.ReLU(inplace=True)
        self.P7_2 = StdConv(feature_size, feature_size, 3, 2, 1)

    def forward(self, x):
        c3, c4, c5 = x
        p5_x = self.P5_1(c5)
        p5_up = self.P5_up(p5_x)
        p5_x = self.P5_2(p5_x)

        p4_x = self.P4_1(c4)
        p4_x = torch.add(p5_up, p4_x)
        p4_up = self.P4_up(p4_x)
        p4_x = self.P4_2(p4_x)

        p3_x = self.P3_1(c3)
        p3_x = torch.add(p3_x, p4_up)
        p6_x = self.P6(c5)
        p7_x = self.P7_1(p6_x)
        p7_x = self.P7_2(p7_x)

        return [p3_x, p4_x, p5_x, p6_x, p7_x]


class RegressionSubNet(nn.Module):
    def __init__(self, in_channel: int, num_anchor=9, feature_size=256):
        super(RegressionSubNet, self).__init__()
        self.conv1 = StdConv(in_channel, feature_size, 3, 1)
        self.conv2 = StdConv(in_channel, feature_size, 3, 1)
        self.conv3 = StdConv(in_channel, feature_size, 3, 1)
        self.conv4 = StdConv(in_channel, feature_size, 3, 1)
        self.output = nn.Conv2d(feature_size, num_anchor*4, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)  # out.shape = B,C,W,H & with 4*num_anchors
        return out.reshape(out.shape[0], -1, 4)


class ClassificationSubNet(nn.Module):
    def __init__(self, in_channel: int, num_anchor=9, num_class=80, feature_size=256):    # prior=0.01,
        super(ClassificationSubNet, self).__init__()

        self.num_class = num_class
        self.num_anchor = num_anchor

        self.conv1 = StdConv(in_channel, feature_size, 3, 1)
        self.conv2 = StdConv(in_channel, feature_size, 3, 1)
        self.conv3 = StdConv(in_channel, feature_size, 3, 1)
        self.conv4 = StdConv(in_channel, feature_size, 3, 1)
        self.output = nn.Conv2d(feature_size, num_anchor*num_class, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.output(out)
        out = self.output_act(out)

        out1 = out.permute(0, 2, 3, 1)  # out.shape = B,C,W,H  with c = n_class * n_anchor

        batch_size, width, height, channel = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor, self.num_class)

        return out2.reshape(x.shape[0], -1, self.num_class)


class Anchor(nn.Module):
    def __init__(self,pyramid_lv=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchor, self).__init__()
        if pyramid_lv is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, img:torch.Tensor) -> torch.Tensor:
        img_shape = img.shape[:2]
        img_shape = np.array(img_shape)
        img_shape = [(img_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchor(base_size = self.sizes[idx], ratios = self.ratios, scales = self.scales)
            print(anchors)
            shifted_anchors = shift_xy(img_shape[idx], self.strides[idx], anchors)
            print(shifted_anchors)
            all_anchor = np.append(all_anchor, shifted_anchors, axis = 0)

        all_anchor = np.expand_dims(all_anchor, axis = 0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor.astype(np.float32))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinaNet(80).to(device)
    model_info(model, 1, 3, 512, 512, device)
    # a = torch.rand(1, 3, 512, 512)
    # b = Anchor()(a)
    # print(b)