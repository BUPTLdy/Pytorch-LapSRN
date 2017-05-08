#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:18:37 2017

@author: ldy
"""

from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import LasSRN
from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Exa')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set()
test_set = get_test_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + 1e-6)) # epsilon=1e-3
    

print('===> Building model')
model = LasSRN()
model_out_path = "model/model_epoch_{}.pth".format(0)
torch.save(model, model_out_path)
#criterion = CharbonnierLoss()
criterion = nn.MSELoss()
print (model)
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()





def train(epoch):
    
    for i in xrange(250):
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            LR, HR_2_target, HR_4_target, HR_8_target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
            
            if cuda:
                LR = LR.cuda()
                HR_2_target = HR_2_target.cuda()
                HR_4_target = HR_4_target.cuda()
                HR_8_target = HR_8_target.cuda()
    
            optimizer.zero_grad()
            HR_2, HR_4, HR_8 = model(LR)
            
            loss1 = CharbonnierLoss(HR_2, HR_2_target)
            loss2 = CharbonnierLoss(HR_4, HR_4_target)
            loss3 = CharbonnierLoss(HR_8, HR_8_target)
            loss = loss1+loss2+loss3   
            
            epoch_loss += loss.data[0]
            loss.backward()
            optimizer.step()
    
            #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
        print("===> Epoch {}, Loop{}: Avg. Loss: {:.4f}".format(epoch, i, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr1 = 0
    avg_psnr2 = 0
    avg_psnr3 = 0
    for batch in testing_data_loader:
        LR, HR_2_target, HR_4_target, HR_8_target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
        
        if cuda:
            LR = LR.cuda()
            HR_2_target = HR_2_target.cuda()
            HR_4_target = HR_4_target.cuda()
            HR_8_target = HR_8_target.cuda()

        HR_2, HR_4, HR_8 = model(LR)
        mse1 = criterion(HR_2, HR_2_target)
        mse2 = criterion(HR_4, HR_4_target)
        mse3 = criterion(HR_8, HR_8_target)        
        psnr1 = 10 * log10(1 / mse1.data[0])
        psnr2 = 10 * log10(1 / mse2.data[0])
        psnr3 = 10 * log10(1 / mse3.data[0])
        avg_psnr1 += psnr1
        avg_psnr2 += psnr2
        avg_psnr3 += psnr3
    print("===> Avg. PSNR1: {:.4f} dB".format(avg_psnr1 / len(testing_data_loader)))
    print("===> Avg. PSNR2: {:.4f} dB".format(avg_psnr2 / len(testing_data_loader)))
    print("===> Avg. PSNR3: {:.4f} dB".format(avg_psnr3 / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

lr=opt.lr
for epoch in range(1, opt.nEpochs + 1):

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-5)
    train(epoch)
    test()
    if epoch % 10 ==0:
        lr = lr/2
        checkpoint(epoch)
