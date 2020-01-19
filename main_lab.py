# @Author: Narsi Reddy <itsnarsi>
# @Date:   2020-01-11T12:27:25-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   cibitaw1
# @Last modified time: 2020-01-18T20:12:09-06:00
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
from utils.data import dataset_1
from utils.trainer_utils import *
from utils.metrics import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


batch_size = 16
base_lr = 5e-4
max_lr = 1e-3

train_src = ['/media/cibitaw1/datasets/superrez/train/mobile',
             '/media/cibitaw1/datasets/superrez/train/prof']
train_dl = dataset_1(train_src)
train_dl = DataLoader(train_dl, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# test_src = '/media/narsi/fast_drive/super_resolution/fliker1k/val'
# test_dl = dataset_1(test_src)
# test_dl = DataLoader(test_dl, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

from utils.trainer import fit_model
model = models.QuantACTShuffleV5()
# stat(model, (3, 32, 32))
# exit()

optimizer = torch.optim.Adam(parfilter(model), base_lr, weight_decay=0.00001)
optimizer_step = None
#torch.optim.lr_scheduler.MultiStepLR(optimizer, [len(train_dl)//2], gamma=0.1)

# from torch_lr_finder import LRFinder
# loss = MSSSIM()#ContentLoss()
# loss = loss.cuda()
# lr_finder = LRFinder(model, optimizer, loss, device="cuda")
# train_dl.dataset.two_out = True
# lr_finder.range_test(train_dl, end_lr=100, num_iter=500)
# lr_finder.plot() # to inspect the loss-learning rate graph
# exit()

fit_model(model,
          train_dl,
          optimizer, optimizer_step,
          MSSSIM(),
          num_epochs = 25, init_epoch = 1,
          log_dir = '/media/cibitaw1/DATA/SP2020/compressACT',
          log_instance = 'QuantACTShuffleV5_exp01',
          use_cuda = True, resume_train = False)
