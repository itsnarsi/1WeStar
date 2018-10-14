# @Author: Narsi Reddy <narsi>
# @Date:   2018-10-13T22:34:55-05:00
# @Last modified by:   narsi
# @Last modified time: 2018-10-13T23:37:36-05:00
import os
import time
import numpy as np
import pyprind

import torch
from torch import optim, nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .train_utils import *
from .metrics import accuracy

def train(model, train_data_loader, optimizer_model, optimizer_center, criterion_softmax, criterion_center, train_batches, loss_weight, cuda):

    totalloss_meter = AverageMeter()
    softmax_meter = AverageMeter()
    center_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    # switch to train mode
    model.train()
    # progress bar init
    bar = pyprind.ProgPercent(train_batches, update_interval=1)
    tic = time.time()
    for batch, (batch_data, batch_target) in enumerate(train_data_loader):

        input_var = tensor_to_torch_var(batch_data, cuda)
        target_var = tensor_to_torch_var(batch_target, cuda, volatile=True)

        # Compute gradient and do optimizer step
        optimizer_model.zero_grad()
        optimizer_center.zero_grad()

        # Compute output
        features = model.features(input_var)
        predictions = model.classify(features)

        # Calculate loss
        softmax_loss = criterion_softmax(output, target_var)
        center_loss = criterion_center(target_var, features)

        total_loss = softmax_loss + loss_weight * center_loss

        total_loss.backward()

        optimizer_model.step()
        optimizer_center.step()

        toc = time.time() - tic
        tic = time.time()

        # Metrics
        totalloss_meter.update(total_loss.data.cpu().item(), batch_data.shape[0])
        softmax_meter.update(softmax_loss.data.cpu().item(), batch_data.shape[0])
        center_meter.update(center_loss.data.cpu().item(), batch_data.shape[0])

        acc1, acc5 = accuracy(predictions, target, topk=(1,5))
        acc1_meter.update(acc1[0], batch_data.shape[0])
        acc5_meter.update(acc5[0], batch_data.shape[0])

        # Update log progress bar
        log_ = ' loss:'+ '{0:4.4f}'.format(train_loss.avg)
        log_ += ' softmax:'+ '{0:4.4f}'.format(softmax_meter.avg)
        log_ += ' center:'+ '{0:4.4f}'.format(center_meter.avg)
        log_ += ' acc1:'+ '{0:4.4f}'.format(acc1_meter.avg)
        log_ += ' acc5:'+ '{0:4.4f}'.format(acc5_meter.avg)
        log_ += ' batch time:'+ '{0:2.3f}'.format(toc)
        bar.update(item_id = log_)

        del output, loss, input_var, target_var, mn

    return totalloss_meter.avg, softmax_meter.avg, center_meter.avg, acc1_meter.avg, acc5_meter.avg


def fit_model(model, train_data, optimizer_model, optimizer_center, loss_weight = 1,
              batch_size = 32, num_epochs = 100, init_epoch = 1, scheduler_setps = [10, 20, 50],
              log_dir = None, log_instance = None, use_cuda = True,
              scheduler = None, resume_train = True):

    # Tensorboard Logger
    if tblog is not None:
        if tbinstance is not None:
            train_logger = SummaryWriter(tblog+'/'+tbinstance+'_train')

    # Best results save model
    best_loss = float('inf')
    is_best = False
    weights_loc = tblog+'/weights/'+tbinstance
    if not os.path.exists(weights_loc):
        os.makedirs(weights_loc)

    # Initializing loss functions
    criterion_center = CenterLoss(model.num_classes, model.feat_dim)
    criterion_softmax = nn.CrossEntropyLoss()

    # Initializing cuda
    device = torch.device("cuda"+ if use_cuda else "cpu")
    if use_cuda:
        model.to(device)
        criterion_center = criterion_center.to(device)
        criterion_softmax = criterion_softmax.to(device)

    # Check and resume training
    check_point_file = weights_loc + '/checkpoint.pth.tar'
    if resume_train:
        if os.path.isfile(check_point_file):
            print("=> loading checkpoint '{}'".format(check_point_file))
            checkpoint = torch.load(check_point_file)
            init_epoch = checkpoint['epoch']+1
            best_loss = checkpoint['Loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_center.load_state_dict(checkpoint['optimizer_center'])
            optimizer_model.load_state_dict(checkpoint['optimizer_model'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(check_point_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(check_point_file))

    # Training data information
    train_batches = len(train_data)//batch_size
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if scheduler_setps is not None:
        scheduler = optim.lr_schedulerself.MultiStepLR(optimizer_softmax, milestones=scheduler_setps, gamma=0.1)

    for epoch in range(init_epoch, num_epochs+1):
        if scheduler_setps is not None:
            scheduler.step()

        totalloss, softmax, center, acc1, acc5 = train(model, train_data_loader, optimizer_model, optimizer_center,
                                                       criterion_softmax, criterion_center, train_batches, loss_weight, use_cuda)

        train_logger.add_scalar('loss', totalloss, epoch)
        train_logger.add_scalar('softmax', softmax, epoch)
        train_logger.add_scalar('center', center, epoch)
        train_logger.add_scalar('acc1', acc1, epoch)
        train_logger.add_scalar('acc5', acc5, epoch)
