import sys
import os
import numpy as np
import time
import argparse

import torch.nn.init
import torch.optim as optim
import uuid
from models import NETWORKNAME
import training
import torch.utils.data as data_utils
import Dataset imoport NETWORKNAME_dataset
from training import train_NETWORKNAME, test_NETWORKNAME
from visdom import Visdom
from tqdm import trange

# Arg parsing
parser = argparse.ArgumentParser(description='NETWORKNAME Implementation')

parser.add_argument('--dataset', type=str, default='ImageNetLoader',
                    help='dataset to train on')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--D', type=int, default=64*64, metavar='N',
                    help='dimension of simulated signal')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-epoch', type=int, default=1, metavar='N',
                    help='wait every epochs')
parser.add_argument('--train-from', type=str, default=None, metavar='M',
                    help='model to train from, if any')
parser.add_argument('--load-data', type=str, default=None,
                    help='load dataset')
parser.add_argument('--savefile', type=str, default='NETWORKNAME',
                    help='save file name')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--uuid', type=str, default=uuid.uuid1(), help='(somewhat) unique identifier for the model/job')
parser.add_argument('--hidden', type=int, default=500, help='hidden states')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='enables verbose logging')

parser.add_argument('--visdom', action='store_true', default=False,
                    help='enables enables visdom as the method to show resaults')
parser.add_argument('--port', type=int, default=8097, help='port that the visdom server is running on')
parser.add_argument('--server', type=str, default="https://localhost", help='server adress of target.')

# determinisim
np.random.seed(0)
torch.manual_seed(0)

global args
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# Visdom Setup
if args.visdom:
    try:
        vis = Visdom(port=args.port, server=args.server)
        assert vis.check_connection(timeout_seconds=3), \
                'no connection could be quickly formed'
     except Exception as e:
        print(e.message)

# determinisim
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.cuda:
    newTensor = torch.cuda.DoubleTensor
else:
    newTensor = torch.DoubleTensor

#parameters
SMALL = 1e-16
weight = 1

#Model Setup
model_cls = NETWORKNAME
model = model_cls()

if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

#Dataset
dataset_test = NETOWRKNAME_dataset('data/test')
dataset_train = NETOWORKNAME_dataset('data/train')

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=args.batch_size, shuffle=False, **kwargs)

if not os.path.isdir('models'):
    os.mkdir('models')

if not os.path.isdir('crash_recovery'):
    os.mkdir('crash_recovery')
# Training / Tracking

trainer = train_NETWORKNAME
validator = test_NETWORKNAME

# Zeroed trackers
train_scores = np.zeros(args.epochs)
validation_scores = np.zeros(args.epochs)
test_scores = np.zeros(args.epochs)
epoch_times = np.zeros(args.epochs)
best_valid = float('inf')


def save_checkpoint(dict, is_best, filename, verbose=False):
    if(is_best):
        torch.save(dict,filename)
        return True
    return False


if args.train_from is not None:
    if os.path.isfile(args.train_from):
        print("=> loading checkpoint '{}'".format(args.train_from))
        checkpoint = torch.load(args.train_from)
        args.start_epoch = checkpoint['epoch']
        best_valid = checkpoint['best_valid']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.train_from, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.train_from))
        sys.exit(1)

try:
    start = time.time()
    t = trange(args.epochs)
    for epoch in t:
        train_scores[epoch] = trainer(train_loader, model, optimizer, epoch, args)
        valid_loss = validator(test_loader, model, epoch, args)
        validation_scores[epoch] = valid_loss
        t.set_postfix(epoch=epoch+1, train_loss=train_scores[epoch], valid_loss=valid_loss)
        epoch_times[epoch] = time.time() - start

        
        is_best = valid_loss < best_valid
        best_valid = min(best_valid, valid_loss)  # do something with best_test / save it

        save_checkpoint({
            'args': args,
            'epoch': epoch + 1,
            'best_valid': best_valid,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'time': epoch_times[epoch],
        }, is_best,
            filename=os.path.join('models', args.savefile))

except KeyboardInterrupt:
    pass

except Exception as e:
    save_checkpoint({
        'args': args,
        'epoch': epoch + 1,
        'best_valid': best_valid,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'time': epoch_times[epoch - 1],
    }, True,
        filename=os.path.join('crash_recovery', args.savefile))
    raise e
