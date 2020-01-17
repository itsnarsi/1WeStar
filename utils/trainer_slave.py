# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-18T20:17:50-06:00
# @Last modified by:   narsi
# @Last modified time: 2020-01-16T21:25:39-06:00
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
    for batch, (input_image) in enumerate(train_data_loader):

        input_image = input_image.cuda()

        # Compute gradient and do optimizer step
        optimizer_model.zero_grad()
        # Compute output
        recon_image, encode_master, encode_student = model(input_image)

        # Calculate loss
        total_loss = 0.01 * criterion(input_image, recon_image) +\
         criterion(encode_master, encode_student)#, target_var
        #, L1E, KLD, CCE, ACC1, ACC2

        total_loss.backward()

        optimizer_model.step()
        if optimizer_step is not None: optimizer_step.step()

        toc = time.time() - tic
        tic = time.time()

        psnr_ = psnr_metric(input_image, recon_image)

        # Metrics
        totalloss_meter.update(total_loss.data.cpu().item(), input_image.shape[0])
        psnr_meter.update(psnr_.data.cpu().item(), input_image.shape[0])
        # Update log progress bar
        log_ = ' loss:'+ '{0:4.4f}'.format(totalloss_meter.avg)
        log_ += ' psnr:'+ '{0:4.4f}'.format(psnr_meter.avg)
        log_ += ' batch time:'+ '{0:2.3f}'.format(toc)
        bar.update(item_id = log_)

        del total_loss, input_image, recon_image, encode_master, encode_student

    return totalloss_meter.avg, psnr_meter.avg


def fit_model(model, train_data_loader,
              optimizer_model, optimizer_step,
              criterion,
              num_epochs = 100, init_epoch = 1,
              log_dir = None, log_instance = None, master_log_instance = None,
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


    # Check and resume training
    weights_loc_master = log_dir+'/weights/'+master_log_instance
    check_point_file = weights_loc_master + '/checkpoint.pth.tar'
    if resume_train:
        if os.path.isfile(check_point_file):
            print("=> loading checkpoint '{}'".format(check_point_file))
            checkpoint = torch.load(check_point_file)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(check_point_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(check_point_file))


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
