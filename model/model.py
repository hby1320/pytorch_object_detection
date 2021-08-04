import torch
import torchvision
from torch.nn import Conv2d, ReLU, BatchNorm2d, Module
from efficientnet_pytorch import EfficientNet #pip install efficientnet_pytorch
from torchsummary import summary

from PIL import Image
from torchvision import transforms as t


class std_conv(Module): #TODO activation if add
    def __init__(self, c_in, c_out, k, st, d=1):
        super().__init__()
        self.conv1 = Conv2d(in_channels = c_in,
                            out_channels = c_out,
                            kernel_size = (k,k),
                            stride = (st,st),
                            dilation = (d,d),
                            )
        self.act = ReLU()
        self.bn1 = BatchNorm2d(num_features = c_out)

    def forword(self, x):
        return self.bn1(self.act(self.conv1(x)))


class depth_wise_Conv2d(Conv2d):
    def __init__(self, ch, ks=3, st=1):
        super().__init__(in_channels = ch,
                         out_channels = ch,
                         kernel_size = (ks,ks),
                         stride = (st,st),
                         padding = (ks//2,ks//2),
                         groups = ch)


class point_wise_conv(Conv2d):
    def __init__(self, c_in, c_out):
        super().__init__(in_channels = c_in,
                         out_channels = c_out,
                         stride = (1,1),
                         kernel_size = (1,1)
                         )




class down_Conv(Conv2d):
    def __init__(self, i_ch, o_ch, k=2, s=2, ):
        super().__init__(in_channels = i_ch,
                        out_channels = o_ch,
                        kernel_size = (k,k),
                        stride = (s,s),
                        padding = (k//2,k//2)
                         )


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

    def __init__(self, backbone=''):


        super(Test_model, self).__init__()
        self.backbone = EfficientNet.from_pretrained(backbone, include_top = False)
        # self.Conv1 = std_conv()


    def forward(self, x):
        x = self.backbone(x)


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

    model = Test_model(model_list[0]).to(device)

    # print(model)
    summary(model,(3,512,512),1,'cpu') ## B
    hook_handles = []
    # for layer in model.modules():
    #     if isinstance(layer, torch.nn.modules.conv.Conv2d):
    #         handle = layer.register_forward_hook(save_output)
    #         hook_handles.append(handle)
    #         print()



    image = Image.open('cat.jpg')
    transform = t.Compose([t.Resize((224, 224)), t.ToTensor()])
    X = transform(image).unsqueeze(dim=0).to(device)
    out = model(X)
