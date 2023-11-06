import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class Upsampler_module(nn.Module):
    def __init__(self, scale, n_feats,):
        super(Upsampler_module, self).__init__()
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=n_feats, out_channels=n_feats*4, kernel_size=3, stride=1, padding=1))
                m.append(nn.PixelShuffle(2))

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x


class invPixelShuffle(nn.Module):

    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)

        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b, -1, y // ratio, x // ratio)

class invUpsampler(nn.Module):
    def __init__(self, scale, n_feats):
        super(invUpsampler, self).__init__()

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(invPixelShuffle(2))
                m.append(nn.Conv2d(in_channels=n_feats*4, out_channels=n_feats, kernel_size=3, stride=1, padding=1))
        else:
            raise NotImplementedError
        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)

class ConvBNReLU2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, act=None, norm=None):
        super(ConvBNReLU2D, self).__init__()

        self.layers = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = None
        self.norm = None
        if norm == 'BN':
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm = torch.nn.InstanceNorm2d(out_channels)
        elif norm == 'GN':
            self.norm = torch.nn.GroupNorm(2, out_channels)
        elif norm == 'WN':
            self.layers = torch.nn.utils.weight_norm(self.layers)
        elif norm == 'Adaptive':
            self.norm = AdaptiveNorm(n=out_channels)

        if act == 'PReLU':
            self.act = torch.nn.PReLU()
        elif act == 'SELU':
            self.act = torch.nn.SELU(True)
        elif act == 'LeakyReLU':
            self.act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        elif act == 'ELU':
            self.act = torch.nn.ELU(inplace=True)
        elif act == 'ReLU':
            self.act = torch.nn.ReLU(True)
        elif act == 'Tanh':
            self.act = torch.nn.Tanh()
        elif act == 'Sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'SoftMax':
            self.act = torch.nn.Softmax2d()

    def forward(self, inputs):

        out = self.layers(inputs)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=4,
        bias=True, norm=None, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):

            modules_body.append(
                ConvBNReLU2D(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias, act=act, norm=norm)
            )

        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = Scale(1)

    def forward(self, x):
        res = self.body(x)
        return res + self.res_scale(x)

class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act,norm, n_resblocks,args):
        super(ResidualGroup, self).__init__()
          modules_body = [
                RCAB(n_feat) for _ in range(n_resblocks)]
        modules_body.append(ConvBNReLU2D(n_feat, n_feat, kernel_size, padding=1, act=act, norm=norm))
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x):
        res = self.body(x)
        return res + self.re_scale(x)
