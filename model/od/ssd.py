import torch
import torch.nn as nn
from model.backbone.vgg16 import VGG16
from utill.utills import model_info
from itertools import product


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
        self.bn1 = nn.BatchNorm2d(512)
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
        self.extract = ExtractModule(1024)



    def forward(self, x:torch.Tensor) -> torch.Tensor:
        conv4_3, conv5_3 = self.backbone(x)
        conv4_3 = self.bn1(conv4_3)
        x2, x3, x4, x5 = self.extract(conv5_3)

        return conv4_3, x2, x3, x4, x5


class ExtractModule(nn.Module):
    def __init__(self, in_ch:int, middle:int, out_ch:int,  st, pad:int) -> torch.Tensor:
        super(ExtractModule, self).__init__()
        self.extract_module_1 = nn.Conv2d(in_ch, middle, 1)
        self.extract_module_2 = nn.Conv2d(middle, out_ch, 3, st, pad)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.extract_module_1(x)
        x = self.extract_module_1(x)
        return x


class loc_conf(nn.Module):
    def __init__(self, num_feature, num_class=21, aspect_ratio=4) -> torch.Tensor:
        super(loc_conf, self).__init__()
        self.loc_layer = nn.Conv2d(num_feature, aspect_ratio*4, 3, 1, 1)
        self.conf_layer = nn.Conv2d(num_feature, num_class * 4, 3, 1, 1)

    def forward(self, x) -> torch.Tensor:
        return self.loc_layer(x), self.conf_layer(x)


class ssd_L2Norm(nn.Module):
    def __init__(self, in_ch, scale=20):
        super(ssd_L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ch))
        self.scale = scale
        self.reset_parameters()  # parameter init
        self.esp = 1e-10

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale) ## weight value == scale

    def forward(self, x) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.esp
        x = torch.div(x, norm)
        weights = self.weight.unsqeeze(0).unsqeeze(2).unsqeeze(3).expand_as(x)
        out = weights * x
        return out


class ssd_default_box(nn.Module):
    def __init__(self, img_size=300, feature_map_size=None, strides=[8, 16, 32, 64, 100, 300],
                 min_size=[30, 60, 111, 162, 213, 264], max_size=[60, 111, 162, 213, 264, 315], aspect_ratios=[]):
        super(ssd_default_box, self).__init__()
        if feature_map_size is None:
            feature_map_size = [38, 19, 10, 5, 3, 1]
        self.img_size = img_size
        self.feature_maps_size = feature_map_size
        self.stride = strides
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios

    def make_dbox_list(self):
        mean = []
        for k, f in enumerate(self.feature_maps_size):  # idx / [38, 19, 10, 5, 3, 1]
            for i, j in product(range(f), repeat=2):
                f_k = self.img_size / self.stride[k]

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
    model = SSD512().to(device)
    dumy = torch.Tensor(1,3,512,512).to(device)
    model_info(model, 1, 3, 300, 300,device)

    # from torchvision.models.detection import ssd300_vgg16
    # from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    #
    # anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    #                                        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
    #                                        steps=[8, 16, 32, 64, 100, 300])
    #
    # ssd = SSD(VGG16,anchor_generator, (512,512), 21)
    # ssd = ssd300_vgg16()
    # model_info(ssd, 1, 3, 300, 300, device, depth=5)
