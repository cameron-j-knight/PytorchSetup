import sys
from argparse import Namespace
import torch.nn.functional as F
import torch

def loss_function(recon_x, x):
    return torch.mean((recon_x - x) ** 2)

def train_NETWORKNAME(train_loader, model, optimizer, epoch, args=Namespace()):
    model.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = None
        if args.cuda:
            data = data.cuda()

        recon_batch = model(data)
        loss = loss_function(recon_batch, data)

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            #print('Epoch: {}    Loss: {}'.format(epoch, loss.data.item()))
            sys.stdout.flush()

    return train_loss / len(train_loader.dataset)

def test_NETWORKNAME(test_loader, model, epoch, args=Namespace()):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(test_loader):
        data = data
        if args.cuda:
            data, label = data.cuda(), label.cuda()

        recon_batch =  model(data)
        loss = loss_function(recon_batch, data)

        test_loss += loss.data.item()

    #print('====> Test loss: {:.3f}'.format(test_loss))
    sys.stdout.flush()
    return test_loss

