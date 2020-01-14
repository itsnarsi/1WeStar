# @Author: Narsi Reddy <itsnarsi>
# @Date:   2020-01-11T13:21:22-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   narsi
# @Last modified time: 2020-01-12T21:43:12-06:00

import torch
from torch import nn
import torch.nn.functional as F
from math import exp
import numpy as np

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
