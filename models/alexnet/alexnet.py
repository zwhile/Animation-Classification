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
matplotlib.use('Agg') # needed due to server xdisplay error
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

    lossPrint = 0.001

    data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cartoon_dataset = datasets.ImageFolder(root='Data/Training'
                                           ,transform=data_transform)
    cartoon_dataset_test = datasets.ImageFolder(root='Data/Testing'
                                           ,transform=test_data_transform)
    dataset_loader = torch.utils.data.DataLoader(cartoon_dataset,
                                                 batch_size=128, shuffle=True,
                                                 num_workers=0)
    testloader = torch.utils.data.DataLoader(cartoon_dataset_test,
                                                 batch_size=128, shuffle=False,
                                                 num_workers=0)
    classes = ('BillyMandy', 'Chowder', 'EdEddEddy', 'Fosters', 'Lazlo')

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize, may not be needed
        npimg = img.numpy()
        plt.imsave("preview.png", np.transpose(npimg, (1, 2, 0)))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(dataset_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    net = Net()
    net.cuda()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lossPrint, momentum=0.9, weight_decay=0.0005)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    file = open('alexnet3_Results.txt','w')

    graphLossEpoch = []
    graphStepEpoch = []
    graphLossIteration = []
    graphStepIteration = []
    fig = plt.figure()
    axes = fig.subplots(nrows=2, ncols=1)

    plt.figure(1)


    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(dataset_loader, 0):
            # get the inputs
            inputs, labels = data


            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels) # use if on CPU
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) # use if on GPU

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print('{}.{}: Loss = {}'.format(epoch+1, i+1, loss.data[0]))
            file.write('{}.{}: Loss = {} \n'.format(epoch+1, i+1, loss.data[0]))

            graphLossIteration.append(loss.data[0])
            graphStepIteration.append(i+(375*epoch))
            n = min(len(graphStepIteration), len(graphLossIteration))

            # print statistics
            running_loss += loss.data[0]
            epoch_loss += loss.data[0]
            if i % 375 == 374:    # print every 375 mini-batches (48000 images)
                running_loss = 0.0
                graphLossEpoch.append(epoch_loss/375)
                graphStepEpoch.append(epoch)
                epoch_loss = 0.0
                torch.save(net.state_dict(), '/workspace/shared/biometrics-project/models/alexnet/{}_epochs_training.ptm'.format(epoch+1))

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
            fig.savefig('loss.png')




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
