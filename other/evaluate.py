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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import alexnet
import squeezenet

def main(argv):

    basedir = os.getcwd()

    test_data_transform = transforms.Compose([
            #transforms.Scale((32,32)),
            #transforms.RandomResizedCrop(224),
            #transforms.RandomResizedCrop(32),
            transforms.Resize((224,224)),
            #transforms.RandomResizedCrop((240,320)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 #std=[0.229, 0.224, 0.225])
    ])

    cartoon_dataset_test = datasets.ImageFolder(root='Data/Testing'
                                           ,transform=test_data_transform)
    testloader = torch.utils.data.DataLoader(cartoon_dataset_test,
                                                 batch_size=32, shuffle=False,
                                                 num_workers=8)
    print('Dataset loaded.')

    classes = sorted(os.listdir(os.path.join(basedir, 'Data', 'Training')))

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    #net = alexnet.Net()
    #net .load_state_dict(torch.load('/workspace/shared/biometrics-project/models/alexnet_0036.ptm'))

    net = squeezenet.Net()
    net .load_state_dict(torch.load('/workspace/shared/biometrics-project/models/squeezenet_0002.ptm'))


    print('Model loaded.')
    net.cuda()
    net.eval()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    #dataiter = iter(dataset_loader)
    '''
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    #file.write('Accuracy of the network on the test images: %d %% \n' % (100 * correct / total))
    '''
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for h, data in enumerate(testloader):
        images, labels = data
        print('{}: {}'.format(h, len(labels)))
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.cuda()).squeeze()
        for i, label in enumerate(labels):
            class_correct[label] += c[i]
            class_total[label] += 1

    print('Average accuracy: {}'.format(100*(sum(class_correct)/sum(class_total))))

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        #file.write('Accuracy of %5s : %2d %% \n' % (classes[i], 100 * class_correct[i] / class_total[i]))
    #file.close()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv)
