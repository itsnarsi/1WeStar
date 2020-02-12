# @Author: Narsi Reddy <itsnarsi>
# @Date:   2020-01-11T11:44:42-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   cibitaw1
# @Last modified time: 2020-02-11T18:59:59-06:00
import torch
torch.manual_seed(29)
from torch import nn
import torch.nn.functional as F

WN_ = lambda x: nn.utils.weight_norm(x)

"""
Single CONV blocks:
"""
class BLOCK_3x3(nn.Module):
    def __init__(
        self, in_ch, out_ch, ker, stride = 1
        ):
        super(BLOCK_3x3, self).__init__()
        self.feat = nn.Sequential(
            nn.ReflectionPad2d(ker//2),
            nn.Conv2d(in_ch, out_ch, ker, stride = stride, bias = True)
            )

    def forward(self, x):
        x = self.feat(x)
        return x


"""
Residual CONV blocks:
"""
class RES_3x3_BLOCK1(nn.Module):
    """
        Residual Block:
            [INPUT] -> 2*[CONV 3x3] -> [OUTPUT] + [INPUT]
    """
    def __init__(
        self, in_ch, out_ch, ker, squeeze = 2, res_scale = 0.25
        ):
        super(RES_3x3_BLOCK1, self).__init__()

        self.skip = in_ch == out_ch
        self.rs = res_scale
        self.feat = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            BLOCK_3x3(in_ch, out_ch//squeeze, ker),
            nn.BatchNorm2d(out_ch//squeeze),
            nn.ReLU(inplace=True),
            BLOCK_3x3(out_ch//squeeze, out_ch, ker),
            )

    def forward(self, x):
        out = self.feat(x)
        if self.skip: out = self.rs * out + x
        return out

class RES_3x3_BLOCK2(nn.Module):
    """
        Residual Block:
            [INPUT] -> 2*[CONV 3x3] -> [OUTPUT] + [INPUT]
    """
    def __init__(
        self, in_ch, out_ch, ker, squeeze = 2, res_scale = 0.25
        ):
        super(RES_3x3_BLOCK2, self).__init__()

        self.skip = in_ch == out_ch
        self.rs = res_scale
        self.feat = nn.Sequential(
            nn.SELU(inplace=True),
            BLOCK_3x3(in_ch, out_ch//squeeze, ker),
            nn.SELU(inplace=True),
            BLOCK_3x3(out_ch//squeeze, out_ch, ker),
            )

    def forward(self, x):
        out = self.feat(x)
        if self.skip: out = self.rs * out + x
        return out
