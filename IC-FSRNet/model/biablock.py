import torch.nn as nn
import torch
from model import common
import torch.nn.functional as F

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()
    def forward(self, bilateral_grid, guidemap):

        device = bilateral_grid.get_device()
        N, C, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N,  1, 1).unsqueeze(3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1,1).unsqueeze(3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1

        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()

        guidemap_guide = torch.cat([hg, wg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3

        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)

        return coeff.squeeze(2)


class ApplyCoeffs_adIllu(nn.Module):
    def __init__(self):
        super(ApplyCoeffs_adIllu, self).__init__()

    def forward(self, coeff, full_res_input):

        res = full_res_input*coeff*(1-full_res_input) + full_res_input
        return res

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class BiaGroup(nn.Module):
    def __init__(self, args):
        super(BiaGroup, self).__init__()
        modules_body = []
        for _ in range(args.n_resblocks):
            modules_body.append(BasicBlock2(args))

        self.tail = common.ConvBNReLU2D(args.n_feats, args.n_feats, kernel_size=3, padding=1, act=args.act)
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x, guidance, grid):
        res = x
        for i in range(len(self.body)):
            res = self.body[i](res, guidance, grid)
        res = self.tail(res)
        return res + self.re_scale(x)

class BasicBlock2(nn.Module):
    def __init__(self, args):
        super(BasicBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=args.n_feats, out_channels=args.n_feats, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=args.n_feats, out_channels=args.n_feats, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

        self.basic1 = common.RCAB(args.n_feats)


        self.slice = Slice()
        self.adjust = nn.Sequential(*[nn.Conv2d(in_channels=args.n_feats, out_channels=args.n_feats, kernel_size=3, stride=1, padding=1), nn.Sigmoid()])
        self.apply_coeffs = ApplyCoeffs_adIllu()
    def forward(self, x, guide, coeffs):
        x = self.basic1(x)
        _, c, h, w = x.shape
        res = self.conv2(self.relu(self.conv1(x)))
        res = F.tanh(res)
        slice_coeffs = self.slice(coeffs, guide)
        slice_coeffs = self.adjust(slice_coeffs)
        out = self.apply_coeffs(slice_coeffs, res)

        out = out + x

        return out
class Biablock(nn.Module):
    def __init__(self, args):
        super(Biablock, self).__init__()
        self.body = BiaGroup(args)
    def forward(self, x, guide, coeffs):
        res = self.body(x, guide, coeffs)

        return res
