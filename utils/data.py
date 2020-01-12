# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-18T20:15:32-06:00
# @Last modified by:   itsnarsi
# @Last modified time: 2020-01-11T21:42:41-06:00
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
    RandomResize(min_scale = 0.125, max_scale = 1.0),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32, 32)),
    transforms.ToTensor()
    ])


class dataset_1(Dataset):
    def __init__(self, src_fldr, transform = transform_v1):
        if type(src_fldr) is not list:
            src_fldr = [src_fldr]
        self.imgs = []
        for sf in src_fldr:
            self.imgs += glob(sf + os.sep + '*.png')
        self.num_samples = len(self.imgs)
        self.transform = transform

    def __getitem__(self, i):
        I = Image.open(self.imgs[i]).convert('RGB')
        if self.transform:
            I = self.transform(I)
        return I#, I

    def __len__(self):
        return self.num_samples
