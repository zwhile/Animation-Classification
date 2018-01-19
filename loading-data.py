# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:56:26 2018

@author: zackw
"""

import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
cartoon_dataset = datasets.ImageFolder(root='data',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(cartoon_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)