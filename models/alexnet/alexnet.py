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

class Net(nn.Module):
    def __init__(self, num_classes=len(os.listdir(os.path.join(os.getcwd(), 'Data', 'Training')))):
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
    print('Classes: {}'.format(classes))

    #def imshow(img):
        #img = img / 2 + 0.5     # unnormalize, may not be needed
        #npimg = img.numpy()
        #plt.imsave("preview.png", np.transpose(npimg, (1, 2, 0)))
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    #dataiter = iter(dataset_loader)
    #images, labels = dataiter.next()

    # show images
    #imshow(torchvision.utils.make_grid(images))

    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    net = Net()
    net.cuda()
    #net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
    pretrained_state = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    model_state = net.state_dict()

    pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
    model_state.update(pretrained_state)
    net.load_state_dict(model_state)
    #net.load_state_dict(torch.load('alexnet_200k_0012_epochs_training.ptm'))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    #lossPrint = 0.01
    lossPrint = 0.001 # because starting from iteration 14
    optimizer = optim.SGD(net.parameters(), lr=lossPrint, momentum=0.9, weight_decay=0.0005)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

    resFile = 'alexnet_Results.txt'
    if os.path.isfile(resFile):
        removeUnfinished(resFile)
        lastEpoch = getLastEpoch(resFile)
        file = open(resFile, 'a')
    else:
        file = open(resFile, 'w')
    #file = open('alexnet_200k_Results.txt','w')

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
        torch.save(net.state_dict(), '/workspace/shared/biometrics-project/models/alexnet_{:04}.ptm'.format(epoch))
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

            #graphLossIteration.append(loss.data[0])
            #graphStepIteration.append(i+(epochSize*epoch))
            #n = min(len(graphStepIteration), len(graphLossIteration))

            # print statistics
            #running_loss += loss.data[0]
            #epoch_loss += loss.data[0]
            #if i % epochSize == (epochSize-1):    # print every 203 mini-batches
                #running_loss = 0.0
                #graphLossEpoch.append(epoch_loss/epochSize)
                #graphStepEpoch.append(epoch)
                #epoch_loss = 0.0
                #torch.save(net.state_dict(), '/workspace/shared/UG-Res-S18/models/big_resnet_{:02}_epochs_training.ptm'.format(epoch+1))

            '''
            fig.suptitle('lr = {}'.format(lossPrint))
            plt.subplot(211)
            n = min(len(graphStepIteration), len(graphLossIteration))
            plt.semilogy(np.array(graphStepIteration)[:n], np.array(graphLossIteration)[:n])
            plt.ylabel('Loss')
            plt.xlabel('# Iterations')
            plt.grid(True)
            fig.tight_layout()
            plt.subplot(212)
            n = min(len(graphStepEpoch), len(graphLossEpoch))
            plt.semilogy(np.array(graphStepEpoch)[:n], np.array(graphLossEpoch)[:n])
            plt.ylabel('Loss')
            plt.xlabel('# Epochs')
            plt.grid(True)
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            fig.savefig('lastresnet_loss.png')
            plt.close(fig)
            '''
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()




    print('Finished Training')
    file.write('Finished training.\n')
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    dataiter = iter(dataset_loader)
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    file.write('Accuracy of the network on the test images: %d %% \n' % (100 * correct / total))
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
        file.write('Accuracy of %5s : %2d %% \n' % (classes[i], 100 * class_correct[i] / class_total[i]))
    file.close()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(sys.argv)
