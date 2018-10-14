# @Author: Narsi Reddy <narsi>
# @Date:   2018-10-13T21:57:45-05:00
# @Last modified by:   narsi
# @Last modified time: 2018-10-13T21:57:46-05:00
import os
from PIL import Image
from torch.utils.data in Dataset
import pandas as pd
import numpy as np

class msceleb1m_dataset(data.Dataset):
    def __init__(self, img_fldr, txt_file, transform=None):
        self.img_fldr = img_fldr
        self.transform = transform
        self.img_list = np.asarray(pd.read_csv(self.txt_file, header = None, sep = ' '), dtype = np.object)

    def __getitem__(self, i):
        img_file = os.path.join(self.img_fldr, self.imgList[i, 0])
        img_target = self.imgList[i, 1]

        img = Image.open(img_file).convert('L')

        if self.transform is not None:
            img = self.transform(img)
        return img, img_target

    def __len__(self):
        return len(self.imgList)
