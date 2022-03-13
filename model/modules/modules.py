import torch.nn as nn
import torch


class StdConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel: int, st: int, padding=1, act='relu', d=1):
        super(StdConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel,
                              stride=st,
                              padding=padding,
                              dilation=d,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.mode = act

        if self.mode == 'swish':
            self.act = nn.SiLU(True)
        else:
            self.act = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthWiseConv2d(nn.Conv2d):
    def __init__(self, in_channel: int, kernel: int, st=1, bs=False):
        super().__init__(in_channels=in_channel,
                         out_channels=in_channel,
                         kernel_size=kernel,
                         stride=st,
                         padding=kernel//2,
                         groups=in_channel,
                         bias=bs
                         )


class DilatedDepthWiseConv2d(nn.Conv2d):
    def __init__(self, in_channel: int, kernel: int, st=1, dilated=1, bs=False):
        super().__init__(in_channels=in_channel,
                         out_channels=in_channel,
                         kernel_size=kernel,
                         stride=st,
                         padding=dilated,
                         groups=in_channel,
                         dilation=dilated,
                         bias=bs
                         )

class PointWiseConv(nn.Conv2d):
    def __init__(self, in_channel: int, out_channel: int, kernel=1, st=1, bs=False):
        super().__init__(in_channels=in_channel,
                         out_channels=out_channel,
                         kernel_size=kernel,
                         padding=kernel//2,
                         stride=st,
                         bias=bs
                         )


class DownConv(nn.Conv2d):
    def __init__(self, in_channel: int, out_channel: int, kernel=2, st=2):
        super().__init__(in_channels=in_channel,
                         out_channels=out_channel,
                         kernel_size=kernel,
                         stride=st,
                         padding=kernel//2,
                         bias=False
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
        self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in // r, kernel_size=1),
                                        nn.SiLU(),
                                        nn.Conv2d(n_in // r, n_in, kernel_size=1),
                                        nn.Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class MCbottle(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, beta: int = 4, alpha: int = 4):
        super(MCbottle, self).__init__()
        self.conv1_1 = PointWiseConv(in_channel=in_ch, out_channel=in_ch * beta)
        self.conv1_2 = DepthWiseConv2d(in_ch * beta, k, 1)
        self.conv1_3 = PointWiseConv(in_channel=in_ch * beta, out_channel=out_ch)
        self.se_block = SEBlock(in_ch * beta, alpha=alpha)
        self.bn = nn.BatchNorm2d(in_ch * beta)
        self.bn1 = nn.BatchNorm2d(in_ch * beta)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(True)
        self.act1 = nn.SiLU(True)
        self.act2 = nn.SiLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.bn(self.conv1_1(x)))
        x1 = self.act1(self.bn1(self.conv1_2(x1)))
        x1 = self.se_block(x1)
        x1 = self.act2(self.bn2(self.conv1_3(x1)))
        x1 = torch.add(x1, x)
        return x1


class ICSPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, beta: int = 2, alpha: int = 4):
        super(ICSPBlock, self).__init__()
        self.bottle_1 = MCbottle(in_ch, in_ch, k, beta, alpha)
        self.bottle_2 = MCbottle(in_ch, in_ch, k, beta, alpha)
        self.pw_conv3 = PointWiseConv(in_channel=in_ch, out_channel=in_ch // 2)
        self.pw_conv4 = PointWiseConv(in_channel=in_ch, out_channel=in_ch // 2)
        self.pw_conv5 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.bn4 = nn.BatchNorm2d(out_ch)
        self.act3 = nn.ReLU(True)
        self.act4 = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [1, 512, 8, 8]
        x1 = self.bottle_1(x)
        x1 = self.bottle_2(x1)
        x2 = self.pw_conv3(x1)
        x3 = self.pw_conv4(x)
        x3 = self.act3(self.bn3(torch.cat([x2, x3], dim=1)))
        x3 = self.act4(self.bn4(self.pw_conv5(x3)))
        return x3


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)


def init_conv_random_normal(module: nn.Module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_conv_kaiming(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class MNBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernal: int,
                 dilated: int):
        super(MNBlock, self).__init__()
        self.DilatedDepthwiseConv = nn.Conv2d(in_ch, in_ch, kernal, 1, dilated, dilated, in_ch, False)
        self.BN = nn.BatchNorm2d(in_ch)
        self.PW1 = nn.Conv2d(in_ch,in_ch*4, 1, 1, 0, 1, 1, False)
        self.ACT1 = nn.SiLU(True)
        self.PW2 = nn.Conv2d(in_ch * 4, out_ch, 1, 1, 0, 1, 1, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.DilatedDepthwiseConv(x)
        x1 = self.BN(x1)
        x1 = self.PW1(x1)
        x1 = self.ACT1(x1)
        x1 = self.PW2(x1)
        x1 = torch.add(x, x1)
        return x1