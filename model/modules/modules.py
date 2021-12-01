import torch.nn as nn
import torch


class StdConv(nn.Module):  # TODO activation if add
    def __init__(self, in_channel: int, out_channel: int, kernel: int, st: int, padding=1, actmode='relu', d=1):
        super(StdConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel,
                              stride=st,
                              padding=padding,
                              dilation=d,
                              bias = False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.mode = actmode

        if self.mode == 'swish':
            self.act = nn.SiLU(inplace = True)
        else:
            self.act = nn.ReLU(inplace = True)


    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x1)))


class DepthWiseConv2d(nn.Conv2d):
    def __init__(self, in_channel: int, kernel: int, st=1):
        super().__init__(in_channels=in_channel,
                         out_channels=in_channel,
                         kernel_size=kernel,
                         stride=st,
                         padding=kernel//2,
                         # padding=dilated_reat,
                         # dilation=dilated_reat,
                         groups=in_channel,
                         bias=False
                         )


class PointWiseConv(nn.Conv2d):
    def __init__(self, in_channel: int, out_channel: int, kernel=1, st=1, bs=False):
        super().__init__(in_channels=in_channel,
                         out_channels=out_channel,
                         kernel_size=kernel,
                         padding = kernel//2,
                         stride=st,
                         bias = bs
                         )


class DownConv(nn.Conv2d):
    def __init__(self, in_channel: int, out_channel: int, kernel=2, st=2):
        super().__init__(in_channels=in_channel,
                         out_channels=out_channel,
                         kernel_size=kernel,
                         stride=st,
                         padding=kernel//2,
                         bias = False
                         )


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


class SEBlock(nn.Module):
    """Squeeze-and-excitation block"""
    def __init__(self, n_in, r=24):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in // r, kernel_size = 1),
                                        nn.SiLU(),
                                        nn.Conv2d(n_in // r, n_in, kernel_size = 1),
                                        nn.Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y