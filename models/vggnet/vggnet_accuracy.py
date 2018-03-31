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
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import time
import alexnet
import squeezenet
import resnet
import inception
import vggnet
import csv

def evalEpoch(imgs, thisNet, inClasses, epochNum):
    classes = inClasses
    net = thisNet
    net.load_state_dict(torch.load('/workspace/shared/biometrics-project/models/vggnet_{:04}.ptm'.format(epochNum)))
    #print('\nModel loaded.')
    net.cuda()
    net.eval()

    itAcc = []

    imgs = iter(imgs)

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    #currTime = time.time()
    for h, data in enumerate(tqdm(imgs)):
        images, labels = data
        #print('{}: {}; {}'.format(epochNum, h, (time.time()-currTime)))
        #currTime = time.time()
        outputs = net(Variable(images.cuda(async=True)))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.cuda(async=True)).squeeze()
        for i, label in enumerate(labels):
            class_correct[label] += c[i]
            class_total[label] += 1

    #print('Average accuracy: {}%\n'.format(int(round(100*(sum(class_correct)/sum(class_total))))))


    itAcc.append(epochNum)
    for i in range(len(classes)):
        #print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        #print('Accuracy of {}: {}%'.format(classes[i], int(round(100 * (class_correct[i] / class_total[i])))))
        itAcc.append(round(100 * (class_correct[i] / class_total[i]), 2))
    itAcc.append(round(100*(sum(class_correct)/sum(class_total)), 2))
    print(itAcc)
    allResults.append(itAcc)
    print('')


allResults = []

def main(argv):

    basedir = os.getcwd()

    test_data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cartoon_dataset_test = datasets.ImageFolder(root='Data/Testing'
                                           ,transform=test_data_transform)
    testloader = torch.utils.data.DataLoader(cartoon_dataset_test,
                                                 batch_size=64, shuffle=False,
                                                 num_workers=8)
    print('\nDataset loaded.')

    classes = sorted(os.listdir(os.path.join(basedir, 'Data', 'Training')))
    firstLine = []
    firstLine.append('epochNum')
    for i in classes:
        firstLine.append(i)
    firstLine.append('average')
    allResults.append(firstLine)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    net = vggnet.Net()
    #net = inception.Net()
    #net = resnet.Net()
    #net = alexnet.Net()
    #net = squeezenet.Net()

    for i in range(0, 14):
        evalEpoch(testloader, net, classes, i)
    with open("vggnet_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(allResults)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv)
