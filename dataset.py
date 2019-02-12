"""
Author: Cameron Knight (cjk1144@rit.edu)
Description: a generic dataloader for a pytorch project
"""

import torch
from torch.utils import data as data_utils

import torchvision

import glob
import os


class Dataset(data_utils.Dataset):
    def __init__(self, root_dir='data/', mode='train', file_format='*'):
        self.files = glob.glob(os.path.join(root_dir, mode, file_format))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x = torch.load(self.files[index])
        return x
