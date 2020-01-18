# @Author: Narsi Reddy <itsnarsi>
# @Date:   2020-01-11T13:21:22-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   narsi
# @Last modified time: 2020-01-17T20:39:38-06:00

import torch
from torch import nn
import torch.nn.functional as F
from math import exp
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# import contextual_loss as cl
# import contextual_loss.fuctional as F


def psnr_metric(T, P, max = 1.0):
    mse = torch.mean((T - P) ** 2)
    return 20 * torch.log10(max / torch.sqrt(mse))

class MSELOSS(nn.Module):
    def __init__(self):
        super(MSELOSS, self).__init__()
    def forward(self, T, P):
        return torch.mean(torch.pow((T - P), 2))

class L1LOSS(nn.Module):
    def __init__(self):
        super(L1LOSS, self).__init__()
    def forward(self, T, P):
        return torch.mean(torch.abs(T - P))

class MSSSIM(nn.Module):
    def __init__(self):
        super(MSSSIM, self).__init__()
        self.ssim = SSIM(data_range=1, size_average=True, channel=3)


    def forward(self, T, P):
        return torch.mean(torch.pow((T - P), 2)) + 0.1 * (1 - self.ssim(T , P ))


from torchvision import models as TM


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = TM.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = requires_grad

    def forward(self, X):
        h = self.slice1(X)
        h = self.slice2(h)
        return h

def normalize_batch(batch):
    # normalize using imagenet mean and std
    batch = batch * 255.0
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.feat = Vgg16()

    def forward(self, T, P):
        pixel_loss = torch.mean((T - P) ** 2)
        # T = normalize_batch(T)
        # P = normalize_batch(P)
        content_loss = torch.mean((self.feat(T) - self.feat(P)) ** 2)
        return pixel_loss + 0.1 * content_loss
