# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:56:26 2018

@author: zackw
"""

import time

import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
import matplotlib
matplotlib.use('Agg') # needed due to server xdisplay error
import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as optim

import torch.utils.model_zoo as model_zoo

#model_urls = {
#    'squeezenet': 'https://download.pytorch.org/models/squeezenet-owt-4df8aa71.pth',
#}

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class Net(nn.Module):

    num_classes = 0

    def __init__(self, num_classes=len(os.listdir(os.path.join(os.getcwd(), 'Data', 'Training')))):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

def main(argv):

    thisDir = '/workspace/shared/biometrics-project/Data'
    #torch.backends.cudnn.benchmark = True

    train_transform = transforms.Compose([
            ##transforms.Resize((224,224)),
            #transforms.RandomCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            #transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    basedir = os.getcwd()

    train_set = datasets.ImageFolder(root=os.path.join(basedir, 'Data', 'Training'),
                                                transform=train_transform)
    test_set = datasets.ImageFolder(root=os.path.join(basedir, 'Data', 'Testing'),
                                                transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=128, shuffle=True,
                                                num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set,
                                                batch_size=64, shuffle=False,
                                                num_workers=0)
    classes = sorted(os.listdir(os.path.join(basedir, 'Data', 'Training')))

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    net = Net()
    net.cuda()
    #net.load_state_dict(model_zoo.load_url(model_urls['squeezenet']))
    pretrained_state = model_zoo.load_url('https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth')
    model_state = net.state_dict()

    pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
    model_state.update(pretrained_state)
    net.load_state_dict(model_state)
    #net.load_state_dict(torch.load('newsqueezenet_0005_epochs_training.ptm'))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    lossPrint = 1e-3
    #lossPrint = 1e-4
    optimizer = optim.SGD(net.parameters(), lr=lossPrint, momentum=0.9, weight_decay=0.0002)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    #file = open('newsqueezenet_Results.txt','w')
    fileName = 'squeezenet_Results.txt'
    if os.path.isfile(fileName):
        file = open(fileName, 'a')
    else:
        file = open(fileName,'w')

    graphLossEpoch = []
    graphStepEpoch = []
    graphLossIteration = []
    graphStepIteration = []
    #fig = plt.figure()
    #axes = fig.subplots(nrows=2, ncols=1)

    #plt.figure(1)

    #epochSize = 383
    lastEpoch = 0

    for epoch in range(9999):  # loop over the dataset multiple times
        #running_loss = 0.0
        #epoch_loss = 0.0
        epoch += lastEpoch
        torch.save(net.state_dict(), '/workspace/shared/biometrics-project/models/squeezenet_{:04}.ptm'.format(epoch))
        #if epoch % 10 == 0 and epoch != 0:
            #lossPrint = lossPrint/10
            #optimizer = optim.SGD(net.parameters(), lr=lossPrint, momentum=0.9, weight_decay=0.0005)
        currTime = time.time()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            #inputs, labels = data


            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels) # use if on CPU
            inputs, labels = Variable(data[0].cuda(async=True)), Variable(data[1].cuda(async=True)) # use if on GPU

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print('{}.{}: Loss = {} (lr = {}, time = {})'.format(epoch+1, i+1, loss.data[0], lossPrint, float(time.time()-currTime)*(1125/3600)))
            currTime = time.time()
            file.write('{}.{}: Loss = {} (lr = {})\n'.format(epoch+1, i+1, loss.data[0], lossPrint))

            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv)
