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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=len(os.listdir(os.path.join(os.getcwd(), 'Data', 'Training')))):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def Net():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    #model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24))
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
    pretrained_state = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
    #pretrained_state = torch.load('/workspace/shared/UG-Res-S18/archs/resnet/resnet_20_epochs_training.ptm')
    model_state = net.state_dict()

    pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
    model_state.update(pretrained_state)
    net.load_state_dict(model_state)
    #net.load_state_dict(torch.load('/workspace/shared/UG-Res-S18/models/resnet_18_epochs_training.ptm'))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    lossPrint = 0.01
    optimizer = optim.SGD(net.parameters(), lr=lossPrint, momentum=0.9, weight_decay=0.0005)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    fileName = 'resnet_Results.txt'
    if os.path.isfile(fileName):
        file = open(fileName, 'a')
    else:
        file = open(fileName,'w')

    lastEpoch = 0

    #epochSize = 163
    for epoch in range(9999):  # loop over the dataset multiple times
        #running_loss = 0.0
        #epoch_loss = 0.0
        epoch += lastEpoch
        torch.save(net.state_dict(), '/workspace/shared/biometrics-project/models/resnet_{:04}.ptm'.format(epoch))
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
