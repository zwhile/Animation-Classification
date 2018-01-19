# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:56:26 2018

@author: zackw
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import matplotlib.pyplot as plt
import numpy as np

data_transform = transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.RandomResizedCrop(),
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
    #plt.imsave("foo.png", np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(dataset_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))