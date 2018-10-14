# @Author: Narsi Reddy <narsi>
# @Date:   2018-10-13T23:38:10-05:00
# @Last modified by:   narsi
# @Last modified time: 2018-10-14T00:13:04-05:00
import torch
torch.manual_seed(29)
from torch import nn
from twrap import layers as L
import torch.nn.functional as F

class mobile_model_1(nn.Module):
    def __init__(self, num_classes, feat_dim = 256, dropout = 0.2):
        super(mobile_model_1, self).__init__()

        #128x128
        self.l1 = L.CONV2D_BLOCK(3, 1, filters = [64], stride=2, padding='same', activation='relu6', use_bias=True,
                                 dropout = dropout, batch_norm = True, dilation = 1, groups = 1, conv_type = 1, scale = 1.0, pool_type = None)

        self.l2 = L.CONV2D_BLOCK(1, 64, filters = [24], stride=1, padding='same', activation='relu6', use_bias=True,
                                 dropout = dropout, batch_norm = True, dilation = 1, groups = 1, conv_type = 1, scale = 1.0, pool_type = None)

        #64x64
        self.mob1 = L.RESNET_BLOCK(3, 24, filters = [24, 24, 24, 24], stride=1, padding='same', activation='relu6', use_bias=True, dropout = dropout,
                                   batch_norm = True, groups = int(24*4), dilation = 1, scale = 1/4.0, resnet_type = 2, pool_type = None)

        self.mob1s = L.CONV2D_BLOCK(3, 24, filters = [36], stride=2, padding='same', activation='relu6', use_bias=True,
                                 dropout = dropout, batch_norm = True, dilation = 1, groups = int(36*4), conv_type = 4, scale = 1/4.0, pool_type = None)

        #32x32
        self.mob2 = L.RESNET_BLOCK(3, 36, filters = [36, 36, 36, 36], stride=1, padding='same', activation='relu6', use_bias=True, dropout = dropout,
                                   batch_norm = True, groups = int(36*4), dilation = 1, scale = 1/4.0, resnet_type = 2, pool_type = None)

        self.mob2s = L.CONV2D_BLOCK(3, 36, filters = [48], stride=2, padding='same', activation='relu6', use_bias=True,
                                 dropout = dropout, batch_norm = True, dilation = 1, groups = int(48*4), conv_type = 4, scale = 1/4.0, pool_type = None)

        #16x16
        self.mob3 = L.RESNET_BLOCK(3, 48, filters = [48, 48, 48, 48], stride=1, padding='same', activation='relu6', use_bias=True, dropout = dropout,
                                   batch_norm = True, groups = int(48*4), dilation = 1, scale = 1/4.0, resnet_type = 2, pool_type = None)

        self.mob3s = L.CONV2D_BLOCK(3, 48, filters = [72], stride=2, padding='same', activation='relu6', use_bias=True,
                                 dropout = dropout, batch_norm = True, dilation = 1, groups = int(72*4), conv_type = 4, scale = 1/4.0, pool_type = None)

        #8x8
        self.mob4 = L.RESNET_BLOCK(3, 72, filters = [72, 72, 72, 72], stride=1, padding='same', activation='relu6', use_bias=True, dropout = dropout,
                                   batch_norm = True, groups = int(72*4), dilation = 1, scale = 1/4.0, resnet_type = 2, pool_type = None)

        self.mob4s = L.CONV2D_BLOCK(3, 72, filters = [72], stride=2, padding='same', activation='relu6', use_bias=True,
                                 dropout = dropout, batch_norm = True, dilation = 1, groups = int(72*4), conv_type = 4, scale = 1/4.0, pool_type = None)

        #4x4
        self.mob5 = L.RESNET_BLOCK(3, 72, filters = [72, 72, 72, 72], stride=1, padding='same', activation='relu6', use_bias=True, dropout = dropout,
                                   batch_norm = True, groups = int(72*4), dilation = 1, scale = 1/4.0, resnet_type = 2, pool_type = None)

        self.mob5s = L.CONV2D_BLOCK(3, 72, filters = [72], stride=2, padding='same', activation='relu6', use_bias=True,
                                 dropout = dropout, batch_norm = True, dilation = 1, groups = int(72*4), conv_type = 4, scale = 1/4.0, pool_type = None)

        #2x2x72
        self.embeded_feat = L.MLP_BLOCK(int(4*72), neurons = [feat_dim], activation = 'l2norm', use_bias=True)
        self.classify = L.MLP_BLOCK(feat_dim, neurons = [num_classes], activation = 'linear', use_bias=True)

    def features(self, x):

        x = self.l1(x)
        x = self.l2(x)

        x = self.mob1(x)
        x = self.mob1s(x)

        x = self.mob2(x)
        x = self.mob2s(x)

        x = self.mob3(x)
        x = self.mob3s(x)

        x = self.mob4(x)
        x = self.mob4s(x)

        x = self.mob5(x)
        x = self.mob5s(x)

        x = L.flatten(x)

        x = self.embeded_feat(x)

        return x

    def forward(self, x):
        return x = self.features(x)
