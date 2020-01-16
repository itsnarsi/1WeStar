# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-18T20:16:34-06:00
# @Last modified by:   narsi
# @Last modified time: 2020-01-15T23:18:15-06:00
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
            PixelUnshuffle(4),
            BLOCK_3x3(in_ch = 48, out_ch = 24, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 24, out_ch = 24, ker = 3, squeeze = 1),
            RES_3x3_BLOCK1(in_ch = 24, out_ch = 24, ker = 3, squeeze = 1),
            BLOCK_3x3(in_ch = 24, out_ch = 64, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3, squeeze = 2),
            BLOCK_3x3(in_ch = 64, out_ch = 24, ker = 3, stride = 1),
            BinTANH()
            )

        self.D = nn.Sequential(
            BLOCK_3x3(in_ch = 24, out_ch = 96, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            BLOCK_3x3(in_ch = 96, out_ch = 48, ker = 3, stride = 1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(4)
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

class QuantACTShuffleV1(nn.Module):
    def __init__(
        self,
        ):
        super(QuantACTShuffleV1, self).__init__()

        self.E = nn.Sequential(
            PixelUnshuffle(4),
            BLOCK_3x3(in_ch = 48, out_ch = 64, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 64, out_ch = 64, ker = 3, squeeze = 2),
            BLOCK_3x3(in_ch = 64, out_ch = 128, ker = 3, stride = 2),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            BLOCK_3x3(in_ch = 128, out_ch = 128, ker = 3, stride = 1),
            BinTANH()
            )

        self.D = nn.Sequential(
            BLOCK_3x3(in_ch = 128, out_ch = 128, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 2),
            nn.Conv2d(128, 96*4, 1),
            nn.PixelShuffle(2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            BLOCK_3x3(in_ch = 96, out_ch = 48, ker = 3, stride = 1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(4)
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



class QuantACTShuffleV2(nn.Module):
    def __init__(
        self,
        ):
        super(QuantACTShuffleV2, self).__init__()

        self.E = nn.Sequential(
            PixelUnshuffle(4),
            BLOCK_3x3(in_ch = 48, out_ch = 96, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            nn.Conv2d(96, 3, 1),
            nn.BatchNorm2d(3),
            QuantCLIP(8),
            # nn.Hardtanh(),
            )

        self.D = nn.Sequential(
            BLOCK_3x3(in_ch = 3, out_ch = 96, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 2),
            BLOCK_3x3(in_ch = 96, out_ch = 48, ker = 3, stride = 1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(4)
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



class QuantACTShuffleV3(nn.Module):
    def __init__(
        self,
        ):
        super(QuantACTShuffleV3, self).__init__()

        self.E = nn.Sequential(
            PixelUnshuffle(4),
            BLOCK_3x3(in_ch = 48, out_ch = 128, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            nn.Conv2d(128, 3, 1),
            nn.BatchNorm2d(3),
            QuantCLIP(8),
            # nn.Hardtanh(),
            )

        self.D = nn.Sequential(
            BLOCK_3x3(in_ch = 3, out_ch = 128, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4),
            BLOCK_3x3(in_ch = 128, out_ch = 48, ker = 3, stride = 1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(4)
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
