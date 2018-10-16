# @Author: Narsi Reddy <narsi>
# @Date:   2018-10-13T22:38:56-05:00
# @Last modified by:   narsi
# @Last modified time: 2018-10-13T22:42:20-05:00
import torch
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value
    Author: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

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

def numpy_to_torch_var(var, cuda, volatile = False):
    if cuda is True:
        var = torch.from_numpy(var).cuda()
    else:
        var = torch.from_numpy(var)

    if volatile is False:
        var.requires_grad_()
    return var

def tensor_to_torch_var(var, cuda, volatile = False):
    if cuda is True:
        var = var.cuda()

    if volatile is False:
        var.requires_grad_()
    return var
