import torch.nn as nn
import torch
from model import common,fenet,refineblock
from model import biablock
import torch.nn.functional as F
class LightNet(nn.Module):
    def __init__(self, args):
        super(LightNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=args.n_feats, kernel_size=1)

    def forward(self, x):
        return x
class GNet(nn.Module):
    def __init__(self, args):
        super(GNet, self).__init__()

        n_feats = 64
        kernel_size = 3
        self.args = args
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=kernel_size, stride=1, padding=1), nn.ReLU(True)])
        self.conv1 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=1, padding=1),
                                     nn.ReLU(True)])

        self.conv2 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=1,
                                               padding=1),
                                     nn.ReLU(True),
                                     nn.AvgPool2d(2)])
        self.conv3 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=1,
                                               padding=1),
                                     nn.ReLU(True),
                                     nn.AvgPool2d(2)

        ])
        self.conv4 = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=1,
                                               padding=1),
                                     nn.ReLU(True),
                                     nn.AvgPool2d(2)])

        self.tail = nn.Sequential(*[nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=kernel_size, stride=1,
                                               padding=1),
                                    nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=kernel_size,
                                              stride=1,
                                              padding=1),
                                    nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=kernel_size,
                                              stride=1,
                                              padding=1),
                                    nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=kernel_size,
                                              stride=1,
                                              padding=1),
                                    ])

    def forward(self, x):

        head = self.head(x)
        guidance = []
        f = self.conv1(head)
        f1 = self.tail[0](f)
        guidance.append(f1)
        f = self.conv2(f)
        f1 = self.tail[1](f)
        guidance.append(f1)
        f = self.conv3(f)
        f1 = self.tail[2](f)
        guidance.append(f1)
        f = self.conv4(f)
        f1 = self.tail[2](f)
        guidance.append(f1)

        return guidance


class LFSRNet(nn.Module):
    def __init__(self, args):
        super(LFSRNet, self).__init__()
        n_blocks = 8
        self.FENet = fenet.FENet(args)
        self.GNet = GNet(args)
        ## light FSRNet
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=args.n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True)])
        self.down1 =  common.invUpsampler(scale=2, n_feats=args.n_feats)
        self.down_stage1 = biablock.Biablock(args)
        self.down2 =  common.invUpsampler(scale=2, n_feats=args.n_feats)
        self.down_stage2 = biablock.Biablock(args)
        self.down3 = common.invUpsampler(scale=2, n_feats=args.n_feats)
        self.down_stage3 = biablock.Biablock(args)

        self.up21 = common.Upsampler_module(scale=2, n_feats=args.n_feats)
        self.up2_stage1 = biablock.Biablock(args)
        self.up22 = common.Upsampler_module(scale=2, n_feats=args.n_feats)

        self.up2_stage2 = biablock.Biablock(args)
        self.up23 = common.Upsampler_module(scale=2, n_feats=args.n_feats)
        self.up2_stage3 = biablock.Biablock(args)
        self.tail =  nn.Conv2d(in_channels=args.n_feats, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.refine_grid = nn.Sequential(*[refineblock.RB(args),refineblock.RB(args),refineblock.RB(args),refineblock.RB(args),refineblock.RB(args)
                                           ])

    def forward(self, x):
        grid = self.FENet(x)
        guidance = self.GNet(x)
        save_x = x
        feature = self.head(x)
        x4 = self.down1(feature)
        grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        inp1 = self.down_stage1(x4, guidance[1], grid)
        x5 = self.down2(inp1)
        grid = grid.view(grid.shape[0], 64, 16, 16)
        grid = self.refine_grid[0](grid, x5)
        grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        inp2 = self.down_stage2(x5, guidance[2], grid)
        x6 = self.down3(inp2)
        grid = grid.view(grid.shape[0], 64, 16, 16)
        grid = self.refine_grid[1](grid, x6)
        grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        x = self.down_stage3(x6, guidance[3], grid)
        inp3 = x + x6
        x = self.up21(inp3)
        grid = grid.view(grid.shape[0], 64, 16, 16)
        grid = self.refine_grid[2](grid, x)
        grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        x = self.up2_stage1(x, guidance[2], grid)
        inp4 = x + x5
        x = self.up22(inp4)
        grid = grid.view(grid.shape[0], 64, 16, 16)
        grid = self.refine_grid[3](grid, x)
        grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        x = self.up2_stage2(x, guidance[1], grid)
        inp5 = x + x4
        x = self.up23(inp5)
        grid = grid.view(grid.shape[0], 64, 16, 16)
        grid = self.refine_grid[4](grid, x)
        grid = grid.view(grid.shape[0], 64, 4, 8, 8)
        res = self.up2_stage3(x, guidance[0], grid)
        sr = self.tail(res) + save_x
        return sr
def make_model(args):
    return LFSRNet(args)
