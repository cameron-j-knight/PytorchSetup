"""
Author: Cameron Knight: cjk1144@rit.edu
Description: a trainer and tester for a generic model
"""

import torch.nn.functional as F
import torch

from argparse import Namespace

def train(train_loader, model, optimizer, args=Namespace()):
    model.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        y_out = model(data)
        loss = torch.mean(y_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()

        
    return train_loss / len(train_loader.dataset)

def test(test_loader, model, args=Namespace()):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        
        if args.cuda:
            data = data.cuda()

        y_out = model(data)
        loss = torch.mean(y_out)

        test_loss += loss.data.item()

    return test_loss / len(test_loader.dataset)

