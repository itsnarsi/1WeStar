# @Author: Narsi Reddy <itsnarsi>
# @Date:   2020-01-11T12:57:14-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   itsnarsi
# @Last modified time: 2020-01-11T13:56:40-06:00

import os
import torch
import shutil

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, storage_loc = ''):
    # https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
    checkpoint = storage_loc + '/checkpoint.pth.tar'
    best_perf = storage_loc + '/model_best.pth.tar'
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best_perf)

def parfilter(model):
    return filter(lambda p: p.requires_grad, model.parameters())
