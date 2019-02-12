"""
Author: Cameron Knight (cjk1144@rit.edu)
Description: a generic model for a pytorch project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Model(nn.Module):
    def __init__(self, x_dim):
        super(Model, self).__init__()

        self.layer = nn.Linear(x_dim, 10)     

    def forward(self, x):
        batch_size = x.size(0)
        x = self.layer(x)
        return x
