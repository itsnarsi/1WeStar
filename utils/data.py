# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-18T20:15:32-06:00
# @Last modified by:   cibitaw1
# @Last modified time: 2020-02-11T19:02:04-06:00
import os
from PIL import Image
import torch
torch.manual_seed(29)
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import pandas as pd
import numpy as np
from glob import glob
np.random.seed(29)
from tqdm import tqdm

from .data_utils import RandomResize

transform_v1=transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    RandomResize(0.25, 2.0, Image.BICUBIC),
    transforms.RandomCrop((64, 64)),
    transforms.ToTensor()
    ])

transform_o=transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    RandomResize(0.25, 2.0, Image.BICUBIC),
    transforms.RandomCrop((64, 64)),
    ])

transform_x = transforms.ToTensor()



class dataset_1(Dataset):
    def __init__(self, src_fldr, transform = transform_v1, two_out = False):
        if type(src_fldr) is not list:
            src_fldr = [src_fldr]
        self.imgs = []
        for sf in src_fldr:
            self.imgs += glob(sf + os.sep + '*.png')
        self.num_samples = len(self.imgs)
        self.transform = transform

        self.two_out = two_out

    def __getitem__(self, i):
        I = Image.open(self.imgs[i]).convert('RGB')
        if self.transform:
            I = self.transform(I)

        return (I, I) if self.two_out else I

    def __len__(self):
        return self.num_samples


class dataset_2(Dataset):
    def __init__(self, src_fldr, two_out = False, convert = "LAB"):
        if type(src_fldr) is not list:
            src_fldr = [src_fldr]
        self.imgs = []
        for sf in src_fldr:
            self.imgs += glob(sf + os.sep + '*.png')
        self.num_samples = len(self.imgs)

        self.two_out = two_out
        self.convert = convert


    def __getitem__(self, i):
        I = Image.open(self.imgs[i]).convert('RGB')
        I = transform_o(I)
        I = I.convert(self.convert)
        I = transform_x(I)

        return (I, I) if self.two_out else I

    def __len__(self):
        return self.num_samples
