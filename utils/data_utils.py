# @Author: Narsi Reddy <itsnarsi>
# @Date:   2020-01-11T14:09:31-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   itsnarsi
# @Last modified time: 2020-01-11T14:09:41-06:00

import torch
from PIL import Image, ImageFilter
import numpy as np

class RandomResize(object):

    def __init__(self, min_scale = 0.8, max_scale = 1.2, interp = Image.BILINEAR):
        self.min = min_scale
        self.max = max_scale
        self.scale_diff = max_scale - min_scale
        self.interp = interp

    def __call__(self, I):

        scale = self.scale_diff * np.random.random_sample() + self.min

        width, height = I.size

        new_width = int(width * scale)
        new_height = int(height * scale)

        return I.resize((new_width, new_height), self.interp)

    def __repr__(self):
        return self.__class__.__name__ + '()'
