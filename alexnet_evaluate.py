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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class Net(nn.Module):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def main(argv):

    lossPrint = 0.01

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
                                                 batch_size=128, shuffle=False,
                                                 num_workers=0)
    classes = ('BillyMandy', 'Chowder', 'EdEddEddy', 'Fosters', 'Lazlo')

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    net = Net()
    net .load_state_dict(torch.load('/workspace/shared/biometrics-project/models/alexnet2_14_epochs.ptm'))
    #net = torch.load('/workspace/shared/biometrics-project/models/alexnet_22_epochs.ptm')['state_dict']
    net.cuda()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    #dataiter = iter(dataset_loader)
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
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.cuda()).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1


    for i in range(5):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        #file.write('Accuracy of %5s : %2d %% \n' % (classes[i], 100 * class_correct[i] / class_total[i]))
    #file.close()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv)
