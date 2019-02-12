"""
Author: Cameron Knight (cjk11144@rit.edu)
Description: generates fake test data for the generic expirament
architecture
"""

import os
import torch

train_dir = 'data/train'
test_dir = 'data/test'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

train_points = 3000
test_points = 300

data_size = 10

[torch.save(torch.rand(data_size), os.path.join(train_dir, "{}.pth".format(i))) for i in range(train_points)]
[torch.save(torch.rand(data_size), os.path.join(test_dir, "{}.pth".format(i))) for i in range(test_points)]
