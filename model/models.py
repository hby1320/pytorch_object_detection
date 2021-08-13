import torch
import torchvision
import torch.onnx
from torch import nn
from efficientnet_pytorch.utils import MemoryEfficientSwish
from efficientnet_pytorch import EfficientNet  # pip install efficientnet_pytorch
from torchsummary import summary as summary
import backbone.resnet50

from PIL import Image
from torchvision import transforms as t
import matplotlib.pyplot as plt


# def autopad(k:int, p=None) -> int:
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


class StdConv(nn.Module):  # TODO activation if add
    def __init__(self, in_channel: int, out_channel: int, kernel: int, st=1, d=1):
        super(StdConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel,
                              stride=st,
                              padding=kernel//2,
                              dilation=d)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x1)))


class DepthWiseConv2d(nn.Conv2d):
    def __init__(self, in_channel: int, kernal: int, st=1):
        super().__init__(in_channels=in_channel,
                         out_channels=in_channel,
                         kernel_size=kernal,
                         stride=st,
                         padding=kernal//2,
                         groups=in_channel
                         )


class PointWiseConv(nn.Conv2d):
    def __init__(self, in_channel: int, out_channel: int, kernel=1, st=1):
        super().__init__(in_channels=in_channel,
                         out_channels=out_channel,
                         kernel_size=kernel,
                         stride=st
                         )


class DownConv(nn.Conv2d):
    def __init__(self, in_channel: int, out_channel: int, kernal=2, st=2):
        super().__init__(in_channels=in_channel,
                         out_channels=out_channel,
                         kernel_size=kernal,
                         stride=st,
                         padding=st//2)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(SeparableConv2d, self).__init__()

        self.depth_wise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.point_wise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        return self.point_wise(self.depth_wise(x1))


#  retina_net 비교군
class RetinaNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = torch
        self.Conv1 = StdConv(32, 64, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3, c4, c5, p6, p7 = self.backbone(x)
        print(f'{c3,c4,c5,p6,p7,}')
        return c3


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

    model = RetinaNet()
    print(model)
    summary(model, (3, 512, 512), device)
    # model = Test_model(model_list[0]).to(device)
    # print(model)

    # model.set_swish(memory_efficient = False)
    # model = retina_net()
    dummy_data = torch.randn(3, 512, 512)
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
    #     def __call__(self, module, module_in, module_out):
    #         self.outputs.append(module_out)
    #
    #     def clear(self):
    #         self.outputs = []
