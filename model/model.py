import torch
import torchvision
import torch.onnx
from torch.nn import Conv2d, ReLU, BatchNorm2d, Module, Sequential
from efficientnet_pytorch.utils import MemoryEfficientSwish
from efficientnet_pytorch import EfficientNet #pip install efficientnet_pytorch
from torchinfo import summary
from PIL import Image
from torchvision import transforms as t
import matplotlib.pyplot as plt






def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



#TODO activation if add
class std_conv(Module):
    def __init__(self, c_in, c_out, k, st=1, p=None, d=1):
        super(std_conv, self).__init__()
        self.conv = Conv2d(in_channels = c_in, out_channels = c_out,kernel_size = k, stride= st, padding = k//2, dilation = d)
        self.bn = BatchNorm2d(c_out)
        # self.act = ReLU(inplace=True)

    def forword(self, x):
        # return self.act(self.bn(self.conv(x)))
        return self.conv(x)

class depth_wise_Conv2d(Conv2d):
    def __init__(self, ch, ks, st=1):
        super().__init__(in_channels = ch,
                         out_channels = ch,
                         kernel_size = ks,
                         stride = (st,st),
                         padding = ks//2,
                         groups = ch
                         )


class point_wise_conv(Conv2d):
    def __init__(self, c_in, c_out):
        super().__init__(in_channels = c_in,
                         out_channels = c_out,
                         stride = (1,1),
                         kernel_size = (1,1)
                         )


class Down_Conv(Conv2d):
    def __init__(self, i_ch, o_ch, k=2, s=2):
        super().__init__(in_channels = i_ch,
                        out_channels = o_ch,
                        kernel_size = (k,k),
                        stride = (s,s),
                        padding = (k//2,k//2)
                         )


class Separable_conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Separable_conv2d, self).__init__()

        self.depth_wise = Sequential(
            Conv2d(in_channels, in_channels, kernel_size, padding= kernel_size//2),
            BatchNorm2d(in_channels),
            ReLU(inplace=True),
        )# inplace 하면 input 으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage 가 좀 좋아짐. 하지만 input 을 없앰.
        self.point_wise = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=(1,1)),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.point_wise(self.depth_wise(x))


## retina_net 비교군
class retina_net(Module):

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained = True)
        self.Conv1 = std_conv(32,64,3,1,1)

    def forward(self, x):
        x = self.backbone(x)
        return x


class Test_model(Module):
    def __init__(self, backbone='', ich=3):
        super(Test_model, self).__init__()
        self.backbone = EfficientNet.from_pretrained(backbone,in_channels = ich, include_top=False)
        self.backbone.set_swish(memory_efficient = False)
        # self.Conv1 = depth_wise_Conv2d(1280,640,3)

        self.Conv1 = Separable_conv2d(1280,640,3)
        self.Conv2 = Separable_conv2d(640, 1280, 3)
        self.Conv3 = Separable_conv2d(1280, 640, 3)
        self.Conv4 = Separable_conv2d(640, 1280, 3)

    def forward(self, x):
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


    class SaveOutput:
        def __init__(self):
            self.outputs = []

        def __call__(self, module, module_in, module_out):
            self.outputs.append(module_out)

        def clear(self):
            self.outputs = []



    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_list = ['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4',
                  'efficientnet-b5','efficientnet-b6','efficientnet-b7']

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


    model1 = Test_model(model_list[0])
    summary(model1,(1,3,512,512))


    print(model1)

    x = torch.rand(1,3,512,512)
    a = model1(x)
    print(a.shape)
    # model = EfficientNet.from_pretrained(model_list[0])
    # inputs = torch.rand(1, 3, 224, 224)
    # model = EfficientNet.from_pretrained('efficientnet-b1')
    # endpoints = model.extract_endpoints(inputs)
    # print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
    # print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
    # print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
    # print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
    # print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
