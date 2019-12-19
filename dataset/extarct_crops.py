# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-17T20:57:03-06:00
# @Last modified by:   narsi
# @Last modified time: 2019-12-17T21:43:54-06:00
import os
from glob import glob
from PIL import Image
from torchvision.transforms import FiveCrop, RandomCrop
from tqdm import tqdm
import numpy as np

src_fldr = '/media/narsi/LargeData/super_rez/DVI2K/DIV2K_train_HR'
dst_fldr = '/media/narsi/fast_drive/super_resolution/dvi2k/train'
imgs = glob(src_fldr + os.sep + '*.png')


for i in tqdm(range(len(imgs))):

    I = Image.open(imgs[i])
    h, w = I.size
    for scale in [1, 2, 3, 4, 8, 16]:
        new_h = h // scale
        new_w = w // scale
        if new_h >= 480 and new_w >= 480:
            I_ = I.copy()
            if scale > 1:
                I_ = I_.resize((new_h, new_w), Image.BICUBIC)
            I_ = FiveCrop(384)(I_)
            for j in range(5):
                I_[j].save(dst_fldr + os.sep + str(i) + '_' + str(scale) + '_' + str(j) + '.png')
