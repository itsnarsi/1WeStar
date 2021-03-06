# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-17T20:57:03-06:00
# @Last modified by:   cibitaw1
# @Last modified time: 2020-01-23T11:30:04-06:00
import os
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np

src_fldr = '/media/cibitaw1/DATA/super_rez/dtd/images'
dst_fldr = '/media/cibitaw1/datasets/superrez/train/dtd'

if not os.path.exists(dst_fldr): os.makedirs(dst_fldr)

imgs = glob(src_fldr + os.sep + '*' + os.sep + '*.jpg')


for i in tqdm(range(len(imgs))):

    I = Image.open(imgs[i])
    h, w = I.size
    # for scale in [1, 2, 3, 4, 8, 16]:
    #     new_h = h // scale
    #     new_w = w // scale
    #     if new_h >= 480 and new_w >= 480:
    #         I_ = I.copy()
    #         if scale > 1:
    #             I_ = I_.resize((new_h, new_w), Image.BICUBIC)
    if h >= 480 and w >= 480:
        I_x = extract_patches_2d(np.uint8(I).copy(), (256,256), max_patches=20, random_state=29)
        for j in range(I_x.shape[0]):
            Image.fromarray(I_x[j]).save(dst_fldr + os.sep + str(i) + '_' + str(j) + '.png')
