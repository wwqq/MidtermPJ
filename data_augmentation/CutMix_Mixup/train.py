#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd 

import models
from utils import progress_bar
from cutout import Cutout
from torch.utils.tensorboard import SummaryWriter
from conf import settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--method', default='cutmix',
                    choices=['cutmix','cutout','mixup','none'])
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--cutmix_prob', default=0.5, type=float, help='probility of CutMix')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

if args.method=='cutout':
    transform_train.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR100(root='~/data', train=True, download=True,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

testset = datasets.CIFAR100(root='~/data', train=False, download=True,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=8)


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('/cfs/cfs-31b43a0b8/personal/qiangwan/hw'):
    os.mkdir('/cfs/cfs-31b43a0b8/personal/qiangwan/hw')
logname = ('/cfs/cfs-31b43a0b8/personal/qiangwan/hw/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

if args.method=='mixup':
    def method_data(x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

if args.method=='cutmix':
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    def method_data(x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        r = np.random.rand(1)
        if alpha > 0 and r < args.cutmix_prob:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

if args.method=='cutout' or args.method=='none':
    def method_data(x, y, alpha=1.0, use_cuda=True):
        return x,y,y,1.0


def method_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


my_train_loss=[]
my_train_acc=[]
my_test_loss=[]
my_test_acc=[]
writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.model, settings.TIME_NOW))
input_tensor = torch.Tensor(1, 3, 32, 32)
   
input_tensor = input_tensor.cuda()
writer.add_graph(net, input_tensor)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = method_data(inputs, targets,
                                                       args.alpha, use_cuda)
        import cv2
        import torch.nn.functional as F
        img = inputs[0]
        #print(img.shape)
        #exit()
        img = F.interpolate(img.reshape(1,3,32,32), size=[224, 224], mode="bilinear").to(torch.float32)[0]
        imgs = img.permute(1,2,0).cpu().numpy()
        imgs = cv2.normalize(imgs, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite('{}.jpg'.format(batch_idx), imgs)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = method_criterion(criterion, outputs, targets_a, targets_b, lam)
        #train_loss += loss.data[0]
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_iter = (epoch - 1) * len(trainloader) + batch_idx + 1           
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('Train/acc', float(1.*correct/total), n_iter)
        my_train_loss.append(train_loss/(batch_idx+1))
        my_train_acc.append(float(1.*correct/total))
        # pd.DataFrame([my_train_loss,my_train_acc],index=['train_loss','train_accuracy']).T.to_csv(args.model+'_'+args.method+'_'+'train.csv')
        # pd.DataFrame([my_train_loss,my_train_acc],index=['train_loss','train_accuracy']).T.to_csv(args.model+'_'+args.method+'_'+'train.csv',mode='a')
    
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # test_loss += loss.data[0]
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        my_test_loss.append(test_loss/(batch_idx+1))
        my_test_acc.append(float(1.*correct/total))
        writer.add_scalar('Test/loss', loss.item(), epoch)
        writer.add_scalar('Test/acc', float(1.*correct/total), epoch)
        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('/cfs/cfs-31b43a0b8/personal/qiangwan/hw/checkpoints'):
        os.mkdir('/cfs/cfs-31b43a0b8/personal/qiangwan/hw/checkpoints')
    torch.save(state, '/cfs/cfs-31b43a0b8/personal/qiangwan/hw/checkpoints/ckpt.t7' + args.name + '_'
               + str(args.seed))

if args.method=='mixup':
    def adjust_learning_rate(optimizer, epoch):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = args.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if args.method=='cutmix':
    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // (args.epoch * 0.5))) * (0.1 ** (epoch // (args.epoch * 0.75)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if args.method=='cutout' or args.method=='none':
    def adjust_learning_rate(optimizer, epoch):
        lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    pd.DataFrame([my_train_loss,my_train_acc],index=['train_loss','train_accuracy']).T.to_csv(args.model+'_'+args.method+'_'+'train.csv')
    pd.DataFrame([my_test_loss,my_test_acc],index=['test_loss','test_accuracy']).T.to_csv(args.model+'_'+args.method+'_'+'test.csv')
    #n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
    
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])
writer.close()
