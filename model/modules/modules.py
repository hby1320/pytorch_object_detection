import torch.nn as nn
import torch


class StdConv(nn.Module):  # TODO activation if add
    def __init__(self, in_channel: int, out_channel: int, kernel: int, st: int, padding=1, d=1):
        super(StdConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel,
                              stride=st,
                              padding=padding,
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
