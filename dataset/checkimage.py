# @Author: Narsi Reddy <narsi>
# @Date:   2020-01-15T06:02:59-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   narsi
# @Last modified time: 2020-01-15T06:05:16-06:00
import os
from glob import  glob
from PIL import  Image


train_src = ['/media/narsi/fast_drive/super_resolution/train/dvi2k',
             '/media/narsi/fast_drive/super_resolution/train/clic_prof',
             '/media/narsi/fast_drive/super_resolution/train/clic_mobile']


imgs = []

for s in train_src:
    imgs += glob(s + os.sep + '*.png')


os.remove("/media/narsi/fast_drive/super_resolution/train/clic_prof/388_28.png")
