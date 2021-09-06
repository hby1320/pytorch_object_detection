import torch
import torch.nn as nn
from model.modules.modules import *
from model.backbone.efficientnetv1 import EfficientNetV1
from utill.utills import model_info

# x[0]: [1, 16, 256, 256]
# x[1]: [1, 24, 128, 128]
# x[2]: [1, 40, 64, 64]
# x[3]:[1, 112, 32, 32]
# x[4]: [1, 320, 16, 16]
# x[5]: [1, 1280, 16, 16]


# class TestModel(nn.Module):
#     def __init__(self, backbone_type:int):
#         super(TestModel, self).__init__()
#         self.backbone = EfficientNetV1(backbone_number = backbone_type)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.backbone(x)
#         return x


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = TestModel(4)
    # model_info(model, 1, 3, 1333, 800, device)

    shifts_x = torch.arange(
        0,  64 * 8, step = 8,
        dtype = torch.float32, device = device
    )
    shifts_y = torch.arange(
        0, 64 * 8, step = 8,
        dtype = torch.float32, device = device
    )
    print(f'{shifts_x=} {shifts_x.size()=}')  #
    print(f'{shifts_y=} {shifts_y.size()=}')

    y, x = torch.meshgrid(shifts_y, shifts_x)

    print(f'{x=} {x.size()=}')  #
    print(f'{y=} {y.size()=}')

    lo_y = y.reshape(-1)
    lo_x = x.reshape(-1)

    print(f'{lo_x=} {lo_x.size()=}')  #
    print(f'{lo_y=} {lo_y.size()=}')
    print(f'{8//2}')
    location =torch.stack((lo_x,lo_y), dim = 1) + 8 //2
    print(f'{location=} {location.size()=}')  #

    # a = torch.tensor([1, 2, 3])
    # b = torch.tensor([4, 5, 6])
    # c, d = torch.meshgrid(b,a)
    # print(f'{c=}\n {d=}')
    #
    # print(f'{c.size()=}\n {d.size()=}')
