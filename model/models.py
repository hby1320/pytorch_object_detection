import torch
import torchvision
import torch.onnx
import torch.nn as nn
from efficientnet_pytorch.utils import MemoryEfficientSwish
from efficientnet_pytorch import EfficientNet  # pip install efficientnet_pytorch
from torchsummary import summary as summary
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
class RetinaNet(nn.Module):

    def __init__(self, num_class: int):
        super().__init__()
        self.backbone = resnet50.ResNet50()
        self.fpn = FeaturePyramid(64, 128, 255)
        self.regression_sub_net = RegressionSubNet(256)
        self.classification_sub_net = ClassificationSubNet(256, num_class=num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x


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
        p5_x = self.p5_2(p5_x)

        p4_x = self.P4_1(c4)
        p4_x = torch.add(p5_up, p4_x)
        p4_up = self.P4_up(p4_x)
        p4_x = self.P4_2(p4_x)

        p3_x = self.p3_1(c3)
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
        return out.contiguous().view(out.shape[0], -1, 4)  # TODO 각 함수 기능 알아보기


class ClassificationSubNet(nn.Module):
    def __init__(self, in_channel, num_anchor=9, num_class=80, feature_size=256):    # prior=0.01,
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

        out2 = out1.view(batch_size, width, height, self.num_anchor, self.num_anchor)

        return out2.contiguous().view(x.shape[0], -1, self.num_class)


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
        print(x.shape)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        print(x.shape)
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
    model = FeaturePyramid(256, 512, 1024).to(device)
    summary(model, (256, 512, 512))

    # model = Test_model(model_list[0]).to(device)
    # print(model)

    # model.set_swish(memory_efficient = False)z
    # model = retina_net()

    # torch.onnx.export(model, dummy_data, "b0.onnx", verbose=True)
    # x = model(dummy_data)
    # print(model)
    # # print(model)
    # # summary(model,(3,512,512),1,'cpu') ## B
    # hook_handles = []
    # save_output = SaveOutput()
    #
    # for layer in model.modules():
    #     if isinstance(layer, torch.nn.modules.conv.Conv2d):
    #         handle = layer.register_forward_hook(save_output)
    #         hook_handles.append(handle)
    #
    #
    #
    #
    # image = Image.open('cat.jpg')
    # transform = t.Compose([t.Resize((224, 224)), t.ToTensor()])
    # X = transform(image).unsqueeze(dim=0).to(device)
    # out = model(X).to(device)
    # plt.figure()
    # plt.imshow(out)
    # model1 = Test_model(model_list[0])
    # summary(model1,(1,3,512,512))
    #
    #
    # print(model1)
    #
    # x = torch.rand(1, 3, 512, 512)
    # a = model1(x)
    # print(a.shape)

    # model = EfficientNet.from_pretrained(model_list[0])
    # inputs = torch.rand(1, 3, 224, 224)
    # model = EfficientNet.from_pretrained('efficientnet-b1')
    # endpoints = model.extract_endpoints(inputs)
    # print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
    # print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
    # print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
    # print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
    # print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])

    # class SaveOutput:
    #     def __init__.py(self):
    #         self.outputs = []
    #
    #     def __call__(self, modules, module_in, module_out):
    #         self.outputs.append(module_out)
    #
    #     def clear(self):
    #         self.outputs = []
