# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-18T20:16:34-06:00
# @Last modified by:   itsnarsi
# @Last modified time: 2020-01-11T21:33:38-06:00
import torch
import numpy as np
torch.manual_seed(29)
from torch import nn
import torch.nn.functional as F
from .model_blocks import *
from .model_utils import *


class MODEL_TEST1(nn.Module):
    def __init__(
        self,
        ):
        super(MODEL_TEST1, self).__init__()

        self.E = nn.Sequential(
            BLOCK_3x3(in_ch = 3, out_ch = 32, ker = 3, stride = 2),
            RES_3x3_BLOCK1(in_ch = 32, out_ch = 32, ker = 3),
            RES_3x3_BLOCK1(in_ch = 32, out_ch = 32, ker = 3),
            BLOCK_3x3(in_ch = 32, out_ch = 64, ker = 3, stride = 2),# BinTANH(),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3),
            BLOCK_3x3(in_ch = 64, out_ch = 128, ker = 3, stride = 2),# BinTANH(),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3),
            BLOCK_3x3(in_ch = 128, out_ch = 256, ker = 3, stride = 2),# BinTANH(),
            RES_3x3_BLOCK1(in_ch = 256, out_ch = 256, ker = 3),
            RES_3x3_BLOCK1(in_ch = 256, out_ch = 256, ker = 3),
            BLOCK_3x3(in_ch = 256, out_ch = 64, ker = 3, stride = 1), BinTANH(),
            )

        self.D = nn.Sequential(
            BLOCK_3x3(in_ch = 64, out_ch = 256, ker = 3, stride = 1),# BinTANH(),
            RES_3x3_BLOCK1(in_ch = 256, out_ch = 256, ker = 3),
            RES_3x3_BLOCK1(in_ch = 256, out_ch = 256, ker = 3),
            nn.UpsamplingNearest2d(scale_factor=2),
            BLOCK_3x3(in_ch = 256, out_ch = 128, ker = 3, stride = 1),# BinTANH(),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3),
            nn.UpsamplingNearest2d(scale_factor=2),
            BLOCK_3x3(in_ch = 128, out_ch = 64, ker = 3, stride = 1),# BinTANH(),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3),
            nn.UpsamplingNearest2d(scale_factor=2),
            BLOCK_3x3(in_ch = 64, out_ch = 64, ker = 3, stride = 1),# BinTANH(),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3),
            nn.UpsamplingNearest2d(scale_factor=2),
            BLOCK_3x3(in_ch = 64, out_ch = 32, ker = 3, stride = 1),# BinTANH(),
            RES_3x3_BLOCK1(in_ch = 32, out_ch = 32, ker = 3),
            RES_3x3_BLOCK1(in_ch = 32, out_ch = 32, ker = 3),
            BLOCK_3x3(in_ch = 32, out_ch = 3, ker = 3, stride = 1), nn.ReLU(inplace=True)
            )

    def encode(self, x):
        # print(x.size())
        x = self.E(x)
        return x

    def decode(self, x):
        # print(x.size())
        x = self.D(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
