#!/usr/bin/env python3
# @Author: Narsi Reddy <cibitlab>
# @Date:   2020-02-11T18:03:26-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   cibitlab
# @Last modified time: 2020-02-11T18:17:52-06:00

import os
import numpy as np

import torch
torch.manual_seed(29)
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
cudnn.benchmark = True

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from glob import glob
from PIL.PngImagePlugin import PngImageFile, PngInfo

def decompress(EnIn, model):

    s_ = 512
    d_ = 128

    w, h = int(EnIn.text['w']), int(EnIn.text['h'])


    I = np.zeros((3,int(s_*np.ceil(h/s_)),int(s_*np.ceil(w/s_))),
                 dtype = np.float32)

    new_en_w = I.shape[2]//4
    new_en_h = I.shape[1]//4


    EnIn = np.uint8(EnIn).copy()
    EnIn = np.pad(EnIn, ((0, int(new_en_h - EnIn.shape[0])),
                         (0, int(new_en_w - EnIn.shape[1])),
                         (0, 0)), mode = "reflect")


    EnIn = np.float32(EnIn)
    EnIn = np.transpose(EnIn, [2, 0, 1])
    for i in range(0, EnIn.shape[1], d_):
        for j in range(0, EnIn.shape[2], d_):
            It = torch.from_numpy(np.expand_dims(EnIn[:, i:i+d_, j:j+d_], 0))/255.0
            It = It * 2.0 - 1.0
            Xe = model(It.cuda())
            I[:, int(i*4):int(i*4)+s_, int(j*4):int(j*4)+s_] = np.clip(Xe.data.squeeze().cpu().numpy(), 0, 1)

    I = np.uint8(255 * I.transpose([1, 2, 0]))

    I = Image.fromarray(I).crop((0, 0, w, h))

    return I

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

"""
Enocder:
"""
class Encoder(nn.Module):
    def __init__(
        self,
        ):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            HaarDWT(3),HaarDWT(12),
            BLOCK_3x3(in_ch = 48, out_ch = 128, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 128, out_ch = 128, ker = 3, squeeze = 4, res_scale = 1.0),
            nn.Conv2d(128, 3, 1),
            QuantCLIP(8)
            )

    def forward(self, x):
        x = self.E(x)
        return x
"""
Deocder:
"""
class Decoder(nn.Module):
    def __init__(
        self,
        ):
        super(Decoder, self).__init__()

        self.D = nn.Sequential(
            BLOCK_3x3(in_ch = 3, out_ch = 256, ker = 3, stride = 1),
            RES_3x3_BLOCK1(in_ch = 256, out_ch = 256, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 256, out_ch = 256, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 256, out_ch = 256, ker = 3, squeeze = 4, res_scale = 1.0),
            RES_3x3_BLOCK1(in_ch = 256, out_ch = 256, ker = 3, squeeze = 4, res_scale = 1.0),
            nn.Conv2d(256, 48, 1),
            HaarIDWT(12),HaarIDWT(3),
            nn.ReLU(),
            )

        self.S = nn.Sequential(nn.ReflectionPad2d(1),
                               nn.AvgPool2d(3, stride=1, padding=0))

    def forward(self, x):
        x = self.D(x)
        x = self.S(x)
        return x

if __name__ == '__main__':


    de_model = Decoder()
    weights_fldr = os.path.dirname(os.path.realpath(__file__)) + os.sep + "decode.pth"
    checkpoint = torch.load(weights_fldr)
    de_model.load_state_dict(checkpoint, strict = False)
    de_model.cuda()
    print('.')


    src_fldr = "/media/cibitaw1/DATA/super_rez/comp_test/decompressed"
    #"./images"

    imgs = glob(src_fldr + os.sep + "*.png")

    for img in imgs:
        I = Image.open(img)
        Iout = decompress(I, de_model)
        Iout.save(img)