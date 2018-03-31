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

import time

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torch.utils.model_zoo as model_zoo

#model_urls = {
#    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
#}

def getEpoch(inLine):
    return int(inLine[:inLine.index('.')])

def getIter(inLine):
    return int(inLine[inLine.index('.')+1:inLine.index(':')])

def getNumIters(inLines):
    assert(getEpoch(inLines[0]) == 1)
    assert(getIter(inLines[0]) == 1)
    cont = True
    ind = 0
    while(cont):
        if getEpoch(inLines[ind]) == 1:
            ind += 1
        else:
            cont = False
            return getIter(inLines[ind-1])

def getLastLine(inLines):
    return inLines[-1]

def removeLines(inLines):
    lastLine = getLastLine(inLines)
    print(lastLine)
    if getIter(lastLine) != getNumIters(inLines):
        newList = [k for k in inLines if getEpoch(k) != getEpoch(lastLine)]
    else:
        newList = inLines
    return newList

def removeUnfinished(thisFile, backup=True):
    from shutil import copyfile
    if backup:
        copyName = thisFile[:thisFile.index('.')]
        copyName += '_backup.txt'
        copyfile(thisFile, copyName)
    if os.path.isfile(thisFile):
        with open(thisFile, 'r') as file:
            lines = file.readlines()
            newList = removeLines(lines)
        with open(thisFile, 'w') as file:
            for i in newList:
                file.write(i)

def getLastEpoch(thisFile):
    with open(thisFile, 'r') as file:
        lines = file.readlines()
        lastLine = getLastLine(lines)
        return int(getEpoch(lastLine))

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=len(os.listdir(os.path.join(os.getcwd(), 'Data', 'Training')))):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def Net(): # densenet161
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24))
    return model

def main(argv):
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
                                                batch_size=32, shuffle=True,
                                                num_workers=8)
    classes = sorted(os.listdir(os.path.join(basedir, 'Data', 'Training')))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    net = Net()
    net = net.cuda()
    #net.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    pretrained_state = model_zoo.load_url('https://download.pytorch.org/models/densenet161-8d451a50.pth')
    model_state = net.state_dict()

    pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
    model_state.update(pretrained_state)
    net.load_state_dict(model_state)
    net.train()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    lossPrint = 0.0001
    optimizer = optim.SGD(net.parameters(), lr=lossPrint, momentum=0.9, weight_decay=0.0005)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    fileName = 'densenet_Results.txt'
    if os.path.isfile(fileName):
        file = open(fileName, 'a')
    else:
        file = open(fileName,'w')

    lastEpoch = 0
    for epoch in range(9999):  # loop over the dataset multiple times
        #running_loss = 0.0
        #epoch_loss = 0.0
        epoch += lastEpoch
        torch.save(net.state_dict(), '/workspace/shared/biometrics-project/models/densenet_{:04}.ptm'.format(epoch))
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
            print('{}.{}: Loss = {} (lr = {}, time = {})'.format(epoch+1, i+1, loss.data[0], lossPrint, round(float(time.time()-currTime)*(4500/3600), 2)))
            currTime = time.time()
            file.write('{}.{}: Loss = {} (lr = {})\n'.format(epoch+1, i+1, loss.data[0], lossPrint))

            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv)
