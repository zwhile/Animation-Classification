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

import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main(argv):
    
    net = Net()
    
    data_transform = transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 #std=[0.229, 0.224, 0.225])
        ])
    cartoon_dataset = datasets.ImageFolder(root='data'
                                           ,transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(cartoon_dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)
    classes = ('BillyMandy', 'Chowder', 'EdEddEddy', 'Fosters', 'Lazlo')
    
    def imshow(img):
        #norm_img = img
        #img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        #print(np.transpose(npimg, (1, 2, 0)))
        print("max unnorm: {}".format(np.amax(np.transpose(npimg, (1, 2, 0)))))
        print("min unnorm: {}".format(np.amin(np.transpose(npimg, (1, 2, 0)))))
        print()
        #npimg = norm_img.numpy()
        #print("max norm: {}".format(np.amax(np.transpose(npimg, (1, 2, 0)))))
        #print("min norm: {}".format(np.amin(np.transpose(npimg, (1, 2, 0)))))
        plt.imsave("foo.png", np.transpose(npimg, (1, 2, 0)))
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    
    # get some random training images
    dataiter = iter(dataset_loader)
    images, labels = dataiter.next()
    
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == "__main__":
    main(sys.argv)