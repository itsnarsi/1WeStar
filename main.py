# @Author: Narsi Reddy <itsnarsi>
# @Date:   2020-01-11T12:27:25-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   narsi
# @Last modified time: 2020-01-13T19:46:06-06:00
import numpy as np

import torch
torch.manual_seed(29)
from torch import nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.parallel
cudnn.benchmark = True
from torch.utils.data import DataLoader

from torchstat import stat
from utils import  models
from utils.trainer import fit_model
from utils.data import dataset_1
from utils.trainer_utils import parfilter
from utils.metrics import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


batch_size = 64
base_lr = 3e-4
max_lr = 1e-3

train_src = ['/media/narsi/fast_drive/super_resolution/dvi2k/train']
train_dl = dataset_1(train_src)
train_dl = DataLoader(train_dl, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# test_src = '/media/narsi/fast_drive/super_resolution/fliker1k/val'
# test_dl = dataset_1(test_src)
# test_dl = DataLoader(test_dl, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

model = models.QuantACTShuffleV1()
# stat(model, (3, 32, 32))
# exit()

optimizer = torch.optim.Adam(parfilter(model), base_lr, weight_decay=0.00001)
optimizer_step = None
# torch.optim.lr_scheduler.MultiStepLR(optimizer, [150*len(train_dl)])

# from torch_lr_finder import LRFinder
# lr_finder = LRFinder(model, optimizer, L1LOSS(), device="cuda")
# train_dl.dataset.two_out = True
# lr_finder.range_test(train_dl, end_lr=100, num_iter=500)
# lr_finder.plot() # to inspect the loss-learning rate graph
# exit()

fit_model(model,
          train_dl,
          optimizer, optimizer_step,
          L1LOSS(),
          num_epochs = 300, init_epoch = 1,
          log_dir = '/media/narsi/LargeData/SP_2020/compressACT',
          log_instance = 'QuantACTShuffleV1_exp02',
          use_cuda = True, resume_train = False)


del model, optimizer

base_lr = 1e-4

model = models.QuantACTShuffleV2()

optimizer = torch.optim.Adam(parfilter(model), base_lr, weight_decay=0.00001)
optimizer_step = None


fit_model(model,
          train_dl,
          optimizer, optimizer_step,
          L1LOSS(),
          num_epochs = 300, init_epoch = 1,
          log_dir = '/media/narsi/LargeData/SP_2020/compressACT',
          log_instance = 'QuantACTShuffleV2_exp01',
          use_cuda = True, resume_train = False)
