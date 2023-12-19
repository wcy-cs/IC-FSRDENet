import torch.nn as nn
from model import refineblock
import torch.nn.functional as F
class FENet(nn.Module):
    def __init__(self, args):
        super(FENet, self).__init__()

        n_feats = 64
        kernel_size = 3
        self.args = args
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=n_feats, kernel_size=kernel_size, stride=1, padding=1), nn.ReLU(True)])

        self.down_stage1 = FENetf(args)
    def forward(self, x):

        x = F.interpolate(x, scale_factor=1/16, mode="bicubic")
        head = self.head(x)
        grid = self.down_stage1(head)
        return grid
class FENetf(nn.Module):
    def __init__(self, args):
        super(FENetf, self).__init__()
        self.args = args
        self.enhance = nn.Sequential(nn.Conv2d(args.n_feats, args.n_feats, 5, stride=1, padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(args.n_feats, args.n_feats, 3, stride=1, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(args.n_feats, args.n_feats, 1, stride=1, padding=0), )
        embed_dim = 8
        self.block = refineblock.SingleFusionTransformer(dim=embed_dim, num_heads=2)
        self.input_proj = refineblock.InputProj(in_channel=args.n_feats, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = refineblock.OutputProj(in_channel=embed_dim, out_channel=args.n_feats, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.input_proj(x)
        x1 = self.block(x, x)
        x = self.output_proj(x1)
        x = self.enhance(x)
        return x

