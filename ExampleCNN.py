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
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2, padding=1)
        self.conv1 = nn.Conv2d(3, 64, 2) # old: (3,6,5) 
        self.conv2 = nn.Conv2d(64, 128, 2) # old: (6,16,5)
        self.conv3 = nn.Conv2d(128, 256, 2) # added by ZW
        self.conv4 = nn.Conv2d(256, 512, 2) # added by ZW
        self.fc1 = nn.Linear(512 * 15 * 20, 1000) # old: 16 * 5 * 5
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100,10)

    def forward(self, x):
        #print("start: {}".format(x.size()))
        x = self.pool(F.relu(self.conv1(x))) 
        #print("conv1: {}".format(x.size()))
        x = self.pool(F.relu(self.conv2(x))) # [4 x ]
        #print("conv2: {}".format(x.size()))
        x = self.pool(F.relu(self.conv3(x)))
        #print("conv3: {}".format(x.size()))
        x = self.pool(F.relu(self.conv4(x)))
        #print("conv4: {}".format(x.size()))
        x = x.view(-1, 512 * 15 * 20) # old: 16 * 5 * 5
        #print("view: {}".format(x.size()))
        x = F.relu(self.fc1(x))
        #print("fc1: {}".format(x.size()))
        x = F.relu(self.fc2(x))
        #print("fc2: {}".format(x.size()))
        x = self.fc3(x)
        #print("fc3: {}".format(x.size()))
        #print()
        return x

def main(argv):
    
    data_transform = transforms.Compose([
            #transforms.Scale((32,32)),
            #transforms.RandomResizedCrop(224),
            #transforms.RandomResizedCrop(32),
            transforms.Resize((240,320)),
            #transforms.RandomResizedCrop((240,320)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_data_transform = transforms.Compose([
            #transforms.Scale((32,32)),
            #transforms.RandomResizedCrop(224),
            #transforms.RandomResizedCrop(32),
            transforms.Resize((240,320)),
            #transforms.RandomResizedCrop((240,320)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))            
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 #std=[0.229, 0.224, 0.225])
    ])
    
    cartoon_dataset = datasets.ImageFolder(root='Data/Training'
                                           ,transform=data_transform)
    cartoon_dataset_test = datasets.ImageFolder(root='Data/Testing'
                                           ,transform=test_data_transform)    
    dataset_loader = torch.utils.data.DataLoader(cartoon_dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=0)
    testloader = torch.utils.data.DataLoader(cartoon_dataset_test,
                                                 batch_size=4, shuffle=False,
                                                 num_workers=0)    
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
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    
    # get some random training images
    dataiter = iter(dataset_loader)
    images, labels = dataiter.next()
    
    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    net = Net()
    net.cuda()
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#----------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------- 
    file = open('testfile.txt','w') 
    
    graphLoss = []
    graphStep = []
    fig = plt.figure()
    n = min(len(graphStep), len(graphLoss))
    plt.semilogy(np.array(graphStep)[:n], np.array(graphLoss)[:n])
    plt.title('Loss')
    plt.grid(True)

    fig.tight_layout()
  #print('file: {}'.format(file[48:]))
    fig.savefig('losspic.png')
    
    
    for epoch in range(1000):  # loop over the dataset multiple times

        running_loss = 0.0
        #print(dataset_loader.__len__())
        for i, data in enumerate(dataset_loader, 0):
            # get the inputs
            inputs, labels = data
    
            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            graphLoss.append(loss.data[0])
            graphStep.append(epoch*len(dataset_loader) + i)
            n = min(len(graphStep), len(graphLoss))
            plt.semilogy(np.array(graphStep)[:n], np.array(graphLoss)[:n])
            plt.title('Loss')
            plt.grid(True)
            fig.tight_layout()
            fig.savefig('losspic.png')
    
            # print statistics
            running_loss += loss.data[0]
            if i == 0 and epoch == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss)) # originally 2000
                file.write('[%d, %5d] loss: %.3f \n' % (epoch + 1, i + 1, running_loss)) # originally 2000
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500)) # originally 2000
                file.write('[%d, %5d] loss: %.3f \n' % (epoch + 1, i + 1, running_loss / 500)) # originally 2000
                running_loss = 0.0
            

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
