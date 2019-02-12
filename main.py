
"""
Author: Cameron Knight (cjk1144@rit.edu)
Description: A generic main file for a pytorch project
"""

import sys
import os
import numpy as np
import argparse
import random
from visdom import Visdom
from tqdm import trange

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils

from torchvision.utils import make_grid, save_image

from model import Model
from dataset import Dataset
from training import train, test

# Arg parsing
parser = argparse.ArgumentParser(description='Model Implementation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batch-size', '-b', type=int, default=64, metavar='N',
                    help='Batch size to use in training.')
parser.add_argument('--dimension', '-D', type=int, default=10, metavar='N',
                    help='Lenght of a single unit of the input data.')
parser.add_argument('--epochs', '-e', type=int, default=500, metavar='N',
                    help='Number of epochs.')
parser.add_argument('--data', type=str, default='data/', metavar='DIR', help="Root folder of data.")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, metavar='N',
                    help='Number to seed randomness from.')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='Continue training from stored checkpoint.')
parser.add_argument('--checkpoint', '-c', type=str, metavar='FILE', default='checkpoint',
                    help='Checkpoint location.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, metavar="FLOAT",
                    help='Learning rate.')
parser.add_argument('--visdom', type=str, metavar='NAME',
                    help='Enables enables visdom logging with NAME.')
parser.add_argument('--port', '-p', type=int, default=8097, metavar='N', help='Port that the visdom server is running on.')
parser.add_argument('--device', '-d', type=int, default=0, metavar="N", help='GPU device number to run on.')
parser.add_argument('--mode', '-m', type=str, default="train_test", 
        choices=['train', 'test', 'train_test', 'results'],
        help='mode to run model (train, test, train_test, results)')

global args
args = parser.parse_args()

# cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()

# determinisim
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

# running mode
args.train = 'train' in args.mode
args.test = 'test' in args.mode
args.results = 'results' in args.mode

args.start_epoch = 0

# Visdom Setup
if args.visdom is not None: 
    vis = Visdom(port=args.port)
else:
    vis = None

# Model Setup
model = Model(args.dimension)
if args.cuda:
    model.cuda(device=args.device)

# Optimizer Setup
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Dataset Setup
dataset_train = Dataset(root_dir=args.data, mode='train', file_format='*.pth')
dataset_test = Dataset(root_dir=args.data, mode='test', file_format='*.pth')

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=args.batch_size, shuffle=False, **kwargs)

# redirect std out
sys.stdout.flush() # <--- important when redirecting to files
newstdout = os.dup(1)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1)
os.close(devnull)
sys.stdout = os.fdopen(newstdout, 'w')

##################
#### VISDOM  #####
##################

losses = {
        'iters': [],
        'loss': [],
        }

def vis_loss_components(dataset, model, vis, sample_size=128):
    global losses
    model.eval()
    l = len(dataset)
    idx = random.randint(0, l-sample_size)
    x = torch.stack([dataset[i].cuda() if args.cuda else dataset[i].cpu()  for i in range(idx, idx + sample_size)], 0) 
    y = model(x)
    loss = torch.mean(y)
    losses['iters'] += [losses['iters'][-1] + 1 if len(losses['iters']) > 0 else 0]
    iters = torch.Tensor(losses['iters'])
    losses['loss'] += [torch.mean(loss).cpu()]
    loss = torch.stack(losses['loss'], 0)
    vis.line(
            X=iters,
            Y=loss,
            env=args.visdom,
            opts=dict(
                    width=800,
                    height=800,
                    xlabel="epoch",
                    ylabel="loss",
                    title='loss')
            )
def vis_epoch(vis, epoch, font_size=64):
    vis.text('<font size="{}">epoch: {}/{}</font>'.format(font_size, epoch, args.epochs), env=args.visdom, 
            opts=dict(
                    width=200,
                    height=200,
                    ))

def vis_results(dataset, model, vis, epoch):
    vis.close(env=args.visdom)
    vis_loss_components(dataset, model, vis)
    vis_epoch(vis, epoch)

############################
#### non-visdom results ####
############################

def results(dataset, model, vis=None, epoch=0):
    # visdom results
    if vis is not None and args.visdom is not None: 
        vis_results(dataset, model, vis, epoch)
    
    # other ways to show results
    return

#####################
#### Checkpoints ####
#####################

args.dir, args.filename = os.path.split(args.checkpoint)

if args.dir != '' and  not os.path.isdir(args.dir):
    os.mkdir(args.dir)

def save_checkpoint(dic, filename):
    torch.save(dic, filename)

if args.resume:
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losses = checkpoint['losses']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.train_from, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        sys.exit(1)

###################
#### Main Loop ####
###################

# Starting Results
results(test_loader.dataset, model, vis, args.start_epoch)

# Finish if in results mode
if args.results:
    exit(0)



train_scores = np.zeros(args.epochs)
validation_scores = np.zeros(args.epochs)
with torch.cuda.device(args.device):
    try:
        t = trange(args.start_epoch, args.epochs)
        for epoch in t:
            if args.train:
                train_scores[epoch] = train(train_loader, model, optimizer, args)
            else:
                train_scores[epoch] = 0

            if args.test:
                validation_scores[epoch] = test(test_loader, model, args)
            else:
                validation_scores[epoch] = 0

            results(test_loader.dataset, model, vis, epoch+1)


            save_checkpoint({
                'args': args,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses,
            },
                filename=args.checkpoint)

    except KeyboardInterrupt:
        pass

    except Exception as e:
            raise(e)
            save_checkpoint({
                'args': args,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses,
            },
                filename= os.path.join(args.dir, "recovered_" + args.filename))
