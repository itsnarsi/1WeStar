# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-18T20:15:32-06:00
# @Last modified by:   narsi
# @Last modified time: 2019-12-18T20:35:40-06:00
import os
from PIL import Image
import torch
torch.manual_seed(29)
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from twrap.transform import RandomBlur, RandomResize
import pandas as pd
import numpy as np
from glob import glob
np.random.seed(29)
from tqdm import tqdm

transform_v1=transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    RandomResize(min_scale = 0.6, max_scale = 2.0),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((172, 96)),
    transforms.ToTensor(),
    ])




class dataset_1(Dataset):
    def __init__(self, src_fldr, transform = transform_v1):
        self.imgs = glob(src_fldr + os.sep + '*.png')
        self.num_samples = len(self.imgs)
        self.transform = transform

    def __getitem__(self, i):
        I = Image.open(self.imgs[i])
        if self.transform:
            I = self.transform(I)
        return I

    def __len__(self):
        return self.num_samples
