# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-18T20:17:50-06:00
# @Last modified by:   narsi
# @Last modified time: 2020-01-15T23:03:55-06:00
import os
import time
import numpy as np
from tqdm import tqdm

import torch
from torch import optim, nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from .trainer_utils import AverageMeter, save_checkpoint
from .metrics import psnr_metric

import pyprind


def train(model, train_data_loader,
          optimizer_model, optimizer_step,
          criterion,
          use_cuda, epoch):

    totalloss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    # switch to train mode
    model.train()
    # progress bar init
    bar = pyprind.ProgPercent(len(train_data_loader), update_interval=1)
    tic = time.time()
    for batch, (batch_data) in enumerate(train_data_loader):

        batch_data = batch_data.cuda()

        # Compute gradient and do optimizer step
        optimizer_model.zero_grad()
        # Compute output
        predictions = model(batch_data)

        # Calculate loss
        total_loss = criterion(batch_data, predictions)#, target_var
        #, L1E, KLD, CCE, ACC1, ACC2

        total_loss.backward()

        optimizer_model.step()
        if optimizer_step is not None: optimizer_step.step()

        toc = time.time() - tic
        tic = time.time()

        psnr_ = psnr_metric(batch_data, predictions)

        # Metrics
        totalloss_meter.update(total_loss.data.cpu().item(), batch_data.shape[0])
        psnr_meter.update(psnr_.data.cpu().item(), batch_data.shape[0])
        # Update log progress bar
        log_ = ' loss:'+ '{0:4.4f}'.format(totalloss_meter.avg)
        log_ += ' psnr:'+ '{0:4.4f}'.format(psnr_meter.avg)
        log_ += ' batch time:'+ '{0:2.3f}'.format(toc)
        bar.update(item_id = log_)

        del total_loss, batch_data, predictions

    return totalloss_meter.avg, psnr_meter.avg


def fit_model(model, train_data_loader,
              optimizer_model, optimizer_step,
              criterion,
              num_epochs = 100, init_epoch = 1,
              log_dir = None, log_instance = None,
              use_cuda = True, resume_train = True):

    # Tensorboard Logger
    if log_dir is not None:
        if log_instance is not None:
            train_logger = SummaryWriter(log_dir+'/'+log_instance+'_train')


    # Best results save model
    best_loss = float('inf')
    is_best = False
    weights_loc = log_dir+'/weights/'+log_instance
    if not os.path.exists(weights_loc):
        os.makedirs(weights_loc)

    # Initializing cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        criterion.to(device)
        model.to(device)

    for epoch in range(init_epoch, num_epochs+1):

        totalloss, psnr_ = train(model, train_data_loader,
                                 optimizer_model, optimizer_step,
                                 criterion,
                                 use_cuda, epoch)


        train_logger.add_scalar('loss', totalloss, epoch)
        train_logger.add_scalar('psnr', psnr_, epoch)


        if totalloss <= best_loss:
            is_best = True
            best_loss = totalloss
        else:
            is_best = False
        save_checkpoint({
            'epoch': epoch,
            'arch': log_instance,
            'state_dict': model.state_dict(),
            'Loss': best_loss,
            'optimizer_model' : optimizer_model.state_dict(),
        }, is_best, storage_loc = weights_loc)
