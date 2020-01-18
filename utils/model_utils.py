# @Author: Narsi Reddy <itsnarsi>
# @Date:   2020-01-11T11:44:07-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   narsi
# @Last modified time: 2020-01-18T13:00:29-06:00
import torch
torch.manual_seed(29)
from torch import nn
import torch.nn.functional as F
import numpy as np

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

class quantclip(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(self, input, quant):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        self.save_for_backward(input)
        c = (input.clamp(min=-1, max =1)+1)/2.0 * quant
        c = 2 * (c.round()/quant) - 1
        return c
    @staticmethod
    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < -1] = 0
        grad_input[input > 1] = 0
        return grad_input, None

class QuantCLIP(torch.nn.Module):

    def __init__(self, num_bits, dtype = torch.cuda.FloatTensor):
        super(QuantCLIP, self).__init__()

        self.quant = 2 ** num_bits - 1
        self.quantclip = quantclip

    def forward(self, input):
        return self.quantclip.apply(input, self.quant)

def getHAARFilters(num_filters):
    LL = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    LH = np.asarray([[-0.5, -0.5], [0.5, 0.5]])
    HL = np.asarray([[-0.5, 0.5], [-0.5, 0.5]])
    HH = np.asarray([[0.5, -0.5], [-0.5, 0.5]])

    DWT = np.concatenate((LL[np.newaxis, ...],
                          LH[np.newaxis, ...],
                          HL[np.newaxis, ...],
                          HH[np.newaxis, ...]))[:, np.newaxis, ...]
    DWT = np.float32(DWT)
    DWT = torch.from_numpy(DWT)

    return DWT.repeat(num_filters, 1, 1, 1)

class HaarDWT(torch.nn.Module):
    def __init__(self, in_ch = 1):
        super(HaarDWT, self).__init__()

        weights = getHAARFilters(in_ch)

        self.conv = nn.Conv2d(in_ch, in_ch * 4, 2, stride=2, bias=False, groups = in_ch)
        self.conv.weight.data = weights
        self.conv.weight.requires_grad = False

    def forward(self, input):
        return self.conv(input)

class HaarIDWT(torch.nn.Module):
    def __init__(self, out_ch = 1):
        super(HaarIDWT, self).__init__()

        weights = getHAARFilters(out_ch)

        self.conv = nn.ConvTranspose2d(out_ch * 4, out_ch, 2, stride=2, bias=False, groups = out_ch)
        self.conv.weight.data = weights
        self.conv.weight.requires_grad = False

    def forward(self, input):
        return self.conv(input)
