# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:56:26 2018

@author: zackw
"""

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
from collections import OrderedDict
import math

import time

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torch.utils.model_zoo as model_zoo

#model_urls = {
#    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
#}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

#def Net(num_classes = len(os.listdir(os.path.join(os.getcwd())))):
def Net(num_classes = len(os.listdir(os.path.join(os.getcwd(), 'Data', 'Training')))):
     # densenet161
    model = VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']), num_classes)
    print('num_classes: {}'.format(num_classes))
    return model

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
    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=128, shuffle=True,
                                                num_workers=8)
    classes = sorted(os.listdir(os.path.join(basedir, 'Data', 'Training')))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    net = Net()
    net.cuda()
    #net.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    pretrained_state = model_zoo.load_url('https://download.pytorch.org/models/vgg11-bbd30ac9.pth')
    model_state = net.state_dict()

    pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
    model_state.update(pretrained_state)
    net.load_state_dict(model_state)
    #net.load_state_dict(torch.load('/workspace/shared/UG-Res-S18/models/vggnet_16_epochs_training.ptm'))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    lossPrint = 0.01
    optimizer = optim.SGD(net.parameters(), lr=lossPrint, momentum=0.9, weight_decay=0.0005)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    fileName = 'vggnet_Results.txt'
    if os.path.isfile(fileName):
        file = open(fileName, 'a')
    else:
        file = open(fileName,'w')

    lastEpoch = 0
    for epoch in range(9999):  # loop over the dataset multiple times
        #running_loss = 0.0
        #epoch_loss = 0.0
        epoch += lastEpoch
        torch.save(net.state_dict(), '/workspace/shared/biometrics-project/models/vggnet_{:04}.ptm'.format(epoch))
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
            print('{}.{}: Loss = {} (lr = {}, time = {})'.format(epoch+1, i+1, loss.data[0], lossPrint, round(float(time.time()-currTime)*(1125/3600), 2)))
            currTime = time.time()
            file.write('{}.{}: Loss = {} (lr = {})\n'.format(epoch+1, i+1, loss.data[0], lossPrint))

            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv)
