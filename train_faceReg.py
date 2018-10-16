# @Author: Narsi Reddy <narsi>
# @Date:   2018-10-15T15:05:02-05:00
# @Last modified by:   narsi
# @Last modified time: 2018-10-16T01:34:23-05:00
import numpy as np

# Pytorch
import torch
torch.manual_seed(29)
from torch import nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.parallel
cudnn.benchmark = True
import torchvision.transforms as transforms

from twrap.utils import model_summary
from utils.train_model import fit_model
from utils.models import *
from utils.data_loader import msceleb1m_dataset

logging_dir = '/media/narsi/VizON/FALL2018/1WeStar/tflog'
model_instance_name = 'FACEMOD_1_E1'

batch_size = 32
nb_epochs = 20

img_fldr = '/media/narsi/fast_drive/ms_celeb_1m_144x144'
img_list = '/media/narsi/fast_drive/MS-Celeb-1M_clean_list.txt'

transform=transforms.Compose([
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
dataset = msceleb1m_dataset(img_fldr, img_list, transform=transform)

model = FACEMOD_1(dataset.num_subs)

model_summary([1, 128, 128], model)
fit_model(model, dataset, loss_weight = 0.01,
          batch_size = batch_size, num_epochs = nb_epochs, init_epoch = 1,
          scheduler_setps = [1, 5, 10], log_dir = logging_dir,
          log_instance = model_instance_name, use_cuda = True, resume_train = False)
