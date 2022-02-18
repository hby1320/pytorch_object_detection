import torch
import torch.nn as nn
from model.backbone.vgg16 import VGG16
from utill.utills import model_info
from itertools import product


class SSDL2Norm(nn.Module):
    def __init__(self, input_channel: int = 512, scale: int = 20):
        super(SSDL2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channel))
        self.scale = scale
        self.reset_params() #  init params
        self.eps = 1E-10

    def reset_params(self):
        ''' 결합된 파라미터의 scale 크기 값으로 초기화 '''
        torch.nn.init.constant_(self.weight, self.scale) #  weight 값을 모두 sacle 값으로 바꿈

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv4_3(512, 38, 38) 크기의 텐서를 정규화 후 계수를 곱하여 계산
        norm = x.pow(2).sum(dim=1, keepdims=True).sqrt()+self.eps  #  torch.Size([1, 1, 38, 38])
        x = torch.div(input=x, other=norm)
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) # 1,512,1,1 -> x와 동일하게 변경
        out = weights * x
        return out


class SSD300(nn.Module):
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
    def __init__(self, num_class: int =21):
        super(SSD300, self).__init__()
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)
        self.backbone = VGG16(True, True)
        # conv4_3 L2norm
        self.l2norm = SSDL2Norm(512, 20)  # conv4_3 ch을 sclae 20으로 초기화
        self.extra_layer1 = ExtractModule(1024, 256, 512, 2, 1)
        self.extra_layer2 = ExtractModule(512, 128, 256, 2, 1)
        self.extra_layer3 = ExtractModule(256, 128, 256)
        self.extra_layer4 = ExtractModule(256, 128, 256)
        # Extra_Layer :
        self.loc_conf_layer1 = LocCofModule(512, num_class, 4)
        self.loc_conf_layer2 = LocCofModule(1024, num_class, 6)
        self.loc_conf_layer3 = LocCofModule(512, num_class, 6)
        self.loc_conf_layer4 = LocCofModule(256, num_class, 6)
        self.loc_conf_layer5 = LocCofModule(256, num_class, 4)
        self.loc_conf_layer6 = LocCofModule(256, num_class, 4)

        self.default_box = SSDDefaultBoxModule(300, [38, 19, 10, 5, 3, 1], [8, 16, 32, 64, 100, 300],
                                               [30, 60, 111, 162, 213, 264], [60, 111, 162, 213, 264, 315]
                                               [[2], [2, 3], [2, 3], [2, 3], [2], [2]])
        self.mask_box_list = self.default_box.make_default_box_list()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, out2 = self.backbone(x)
        out1 = self.l2norm(out1)
        out3 = self.extra_layer1(out2)
        out4 = self.extra_layer2(out3)
        out5 = self.extra_layer3(out4)
        out6 = self.extra_layer4(out5)
        out1 = self.loc_conf_layer1(out1)
        out2 = self.loc_conf_layer2(out2)
        out3 = self.loc_conf_layer3(out3)
        out4 = self.loc_conf_layer4(out4)
        out5 = self.loc_conf_layer5(out5)
        out6 = self.loc_conf_layer6(out6)
        return out1, out2, out3, out4, out5, out6


class ExtractModule(nn.Module):
    def __init__(self, in_channel: int, mid_channel: int, out_channel: int, st: int = 1, pad: int = 0):
        super(ExtractModule, self).__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.stride = st
        self.padding = pad
        self.extract_module_1 = nn.Conv2d(self.in_channel, self.mid_channel, 1)
        self.extract_module_2 = nn.Conv2d(self.mid_channel, self.out_channel, 3, self.stride, self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_module_1(x)
        x = self.extract_module_2(x)
        return x


class LocCofModule(nn.Module):
    def __init__(self, num_feature, num_class=21, aspect_ratio: int=4) -> torch.Tensor:
        super(LocCofModule, self).__init__()
        self.loc_layer = nn.Conv2d(num_feature, aspect_ratio*4, 3, 1, 1)
        self.conf_layer = nn.Conv2d(num_feature, num_class * 4, 3, 1, 1)

    def forward(self, x) -> torch.Tensor:
        return self.loc_layer(x), self.conf_layer(x)


class SSDDefaultBoxModule(nn.Module):
    def __init__(self, img_size=300, feature_map_size=None, steps=[8, 16, 32, 64, 128, 256],
                 min_size=[30, 60, 111, 162, 213, 264], max_size=[60, 111, 162, 213, 264, 315], aspect_ratios=[]):
        super(SSDDefaultBoxModule, self).__init__()
        if feature_map_size is None:
            feature_map_size = [38, 19, 10, 5, 3, 1]
        self.img_size = img_size
        self.feature_maps_size = feature_map_size
        self.num_priors = len(feature_map_size)
        self.steps = steps
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios

    def make_default_box_list(self):
        mean = []
        for k, f in enumerate(self.feature_maps_size):  # idx / [38, 19, 10, 5, 3, 1]
            for i, j in product(range(f), repeat=2):
                f_k = self.img_size / self.steps[k]
                # SSD paepr Cx Cy
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                s_k = self.min_size[k]/self.img_size
                mean += [cx, cy, s_k, s_k]

                s_k_prime = torch.sqrt(s_k*(self.max_size[k]/self.img_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                for aspect in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * torch.sqrt(aspect), s_k / torch.sqrt(aspect)]
                    mean += [cx, cy, s_k / torch.sqrt(aspect), s_k * torch.sqrt(aspect)]
        output = torch.Tensor(mean).view(-1, 4).clip(max=1, min=0)
        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSD300().to(device)
    dumy = torch.Tensor(1, 3, 300, 300).to(device)
    model_info(model, 1, 3, 300, 300,device)
