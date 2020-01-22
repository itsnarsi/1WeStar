# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-18T20:16:34-06:00
# @Last modified by:   narsi
# @Last modified time: 2020-01-22T17:48:27-06:00
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

class RBS(nn.Module):
    def __init__(
        self, in_ch = 96, out_ch = 48, squeeze = 3,
        ):
        super(RBS, self).__init__()

        self.SET_1 = nn.Sequential(
            RES_3x3_BLOCK1(in_ch = in_ch, out_ch = in_ch, ker = 3, squeeze = squeeze, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = in_ch, out_ch = in_ch, ker = 3, squeeze = squeeze, res_scale = 1.0),
            )
        self.SET_2 = nn.Sequential(
            RES_3x3_BLOCK1(in_ch = in_ch, out_ch = in_ch, ker = 3, squeeze = squeeze, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = in_ch, out_ch = in_ch, ker = 3, squeeze = squeeze, res_scale = 1.0),
            )

        self.SQEZ = nn.Conv2d(in_ch*2, out_ch, 1)

    def forward(self, x):

        x1 = self.SET_1(x)
        x2 = self.SET_2(x1)

        x = torch.cat((x1, x2), dim = 1)

        return self.SQEZ(x)


class QuantACTShuffleV3(nn.Module):
    def __init__(
        self,
        ):
        super(QuantACTShuffleV3, self).__init__()

        self.E = nn.Sequential(
            PixelUnshuffle(4),
            BLOCK_3x3(in_ch = 48, out_ch = 96, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 3, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 3, res_scale = 1.0),
            nn.Conv2d(96, 3, 1),
            QuantCLIP(8)
            )

        self.D = nn.Sequential(
            nn.Conv2d(3, 96, 1),
            RBS(in_ch = 96, out_ch = 48, squeeze = 3),
            RBS(in_ch = 48, out_ch = 48, squeeze = 2),
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



class QuantACTShuffleV4(nn.Module):
    def __init__(
        self,
        ):
        super(QuantACTShuffleV4, self).__init__()

        self.E = nn.Sequential(
            PixelUnshuffle(4),
            BLOCK_3x3(in_ch = 48, out_ch = 96, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 3, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 3, res_scale = 1.0),
            nn.Conv2d(96, 4, 1),
            QuantCLIP(8)
            )

        self.D = nn.Sequential(
            nn.Conv2d(4, 96, 1),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 3, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 3, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 3, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 3, res_scale = 1.0),
            nn.Conv2d(96, 48, 1),
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



class CleanImg(nn.Module):
    def __init__(
        self, compress_model
        ):
        super(CleanImg, self).__init__()

        self.M = compress_model

        self.E = nn.Sequential(
            BLOCK_3x3(in_ch = 3, out_ch = 48, ker = 3, stride = 2),
            RES_3x3_BLOCK1(in_ch = 48, out_ch = 48, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 48, out_ch = 48, ker = 3, squeeze = 4, res_scale = 1.0),
            BLOCK_3x3(in_ch = 48, out_ch = 12, ker = 3, stride = 1),
            nn.PixelShuffle(2)
            )

    def encode(self, x):
        # print(x.size())
        x = self.M.E(x)
        return x

    def forward(self, x):
        x = self.M(x)
        x = self.E(x) + x
        return x



class QuantACTShuffleV5(nn.Module):
    def __init__(
        self,
        ):
        super(QuantACTShuffleV5, self).__init__()

        self.E = nn.Sequential(
            HaarDWT(3),HaarDWT(12),
            BLOCK_3x3(in_ch = 48, out_ch = 96, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 96, out_ch = 96, ker = 3, squeeze = 4, res_scale = 1.0),
            nn.Conv2d(96, 3, 1),
            QuantCLIP(8)
            )

        self.D = nn.Sequential(
            BLOCK_3x3(in_ch = 3, out_ch = 128, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4, res_scale = 1.0),
            nn.Conv2d(128, 48, 1),
            HaarIDWT(12),HaarIDWT(3),
            nn.ReLU(),
            )

    def encode(self, x):
        x = self.E(x)
        return x

    def decode(self, x):
        x = self.D(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class QuantACTShuffleV6(nn.Module):
    def __init__(
        self,
        ):
        super(QuantACTShuffleV6, self).__init__()

        self.E = nn.Sequential(
            HaarDWT(3),HaarDWT(12),
            BLOCK_3x3(in_ch = 48, out_ch = 128, ker = 3, stride = 1),
            RES_3x3_BLOCK2(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK2(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4, res_scale = 1.0),
            nn.Conv2d(128, 3, 1),
            QuantCLIP(8)
            )

        self.D = nn.Sequential(
            BLOCK_3x3(in_ch = 3, out_ch = 256, ker = 3, stride = 1),
            RES_3x3_BLOCK2(in_ch = 256, out_ch = 256, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK2(in_ch = 256, out_ch = 256, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK2(in_ch = 256, out_ch = 256, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK2(in_ch = 256, out_ch = 256, ker = 3, squeeze = 4, res_scale = 1.0),
            nn.Conv2d(256, 48, 1),
            HaarIDWT(12),HaarIDWT(3),
            nn.ReLU(),
            )

    def encode(self, x):
        x = self.E(x)
        return x

    def decode(self, x):
        x = self.D(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
