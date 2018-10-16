# @Author: Narsi Reddy <narsi>
# @Date:   2018-10-13T21:57:45-05:00
# @Last modified by:   narsi
# @Last modified time: 2018-10-15T15:46:46-05:00
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class msceleb1m_dataset(Dataset):
    def __init__(self, img_fldr, txt_file, transform=None):
        self.img_fldr = img_fldr
        self.transform = transform
        self.img_list = np.asarray(pd.read_csv(txt_file, header = None, sep = ' '), dtype = np.object)
        self.num_subs = len(np.unique(self.img_list[:, 1]))

    def __getitem__(self, i):
        img_file = os.path.join(self.img_fldr, self.img_list[i, 0])
        img_target = self.img_list[i, 1]

        img = Image.open(img_file).convert('L')

        if self.transform is not None:
            img = self.transform(img)
        return img, img_target

    def __len__(self):
        return len(self.img_list)
