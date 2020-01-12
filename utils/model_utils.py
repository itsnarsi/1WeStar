# @Author: Narsi Reddy <itsnarsi>
# @Date:   2020-01-11T11:44:07-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   itsnarsi
# @Last modified time: 2020-01-11T19:57:28-06:00
import torch
torch.manual_seed(29)
from torch import nn
import torch.nn.functional as F


def pixel_unshuffle(x, factor = 2):
    b, c, h, w = x.size()
    new_c = c * factor**2
    new_h = h // factor
    new_w = w // factor
    x = x.contiguous().view(b, c, new_h, factor, new_w, factor)
    x = x.permute(0,1,3,5,2,4).contiguous().view(b, new_c, new_h, new_w)
    return x

class PixelUnshuffle(torch.nn.Module):
    def __init__(self, factor = 2):
        super(PixelUnshuffle, self).__init__()
        self.factor = factor

    def forward(self, input):
        return pixel_unshuffle(input, factor = self.factor)


class binTanH(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        c = input.clamp(min=-1, max =1)
        c[c > 0.0] = 1.0
        c[c <= 0.0] = -1.0
        return c
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < -1] = 0.0
        grad_input[input > 1] = 0.0
        return grad_input, None

class BinTANH(torch.nn.Module):
    def __init__(self):
        super(BinTANH, self).__init__()
        self.quantclip = binTanH

    def forward(self, input):
        return self.quantclip.apply(input)
