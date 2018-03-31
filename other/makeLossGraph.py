#!/usr/bin/python

import sys
import os
import matplotlib
matplotlib.use('Agg') # needed due to server xdisplay error
import matplotlib.pyplot as plt
import numpy as np


def getEpoch(inLine):
    return int(inLine[:inLine.index('.')])

def getIter(inLine):
    return int(inLine[inLine.index('.')+1:inLine.index(':')])

def removeEpoch(inList, epNum):
    retList = []
    for i in inList:
        if getEpoch(i) != epNum:
            retList.append(i)
    return retList

def makeGraph(fileName, inPE):
    os.chdir('/workspace/shared/biometrics-project/')
    print('')
    epochs = []
    perEpoch = inPE # hand-coded unfortunately
    print('For {}:'.format(fileName))
    file = open(fileName, 'r')
    pureLines = file.readlines()
    lastLine = pureLines[len(pureLines)-1]
    lastLine = lastLine[:len(lastLine)-2]
    numEpochs = getEpoch(lastLine)
    numIter = getIter(lastLine)
    #print('before: {}'.format(len(pureLines)/perEpoch))
    if getIter(lastLine) != perEpoch:
        pureLines = removeEpoch(pureLines, getEpoch(lastLine))
        numEpochs -= 1
    #print('after: {}'.format(len(pureLines)/perEpoch))
    epochs = np.zeros((perEpoch,))
    allEpochs = np.zeros((numEpochs, perEpoch))
    #print('lastLine: -{}-'.format(lastLine))
    #print('numEpochs: -{}-'.format(numEpochs))
    #print('numIter: -{}-'.format(numIter))
    #print(numEpochs)
    #print(numEpochs[:numEpochs.index('.')])
    means = np.zeros((numEpochs,))
    epochValues = []
    for i, thisLine in enumerate(pureLines):
        ln = thisLine
        ln = ln[ln.index('L'):]
        ln = ln.replace('Loss = ', '')
        ln = ln[:ln.index(' ')]
        #print('-{}-'.format(ln))
        #ln = ln.replace(' \n', '')
        #print('-{}-'.format(ln))
        pureLines[i] = float(ln)
        allEpochs[getEpoch(thisLine)-1,getIter(thisLine)-1] = ln
        #print('allEpochs[{},{}]: {}'.format(getEpoch(thisLine)-1,
                #getIter(thisLine)-1, allEpochs[getEpoch(thisLine)-1,getIter(thisLine)-1]))
    for i in range(numEpochs):
        means[i] = np.mean(allEpochs[i,:])
        print('means[{}]: {}'.format(i+1, means[i]))
    fig = plt.figure()
    plt.figure(1)
    #fig.suptitle('lr = {}'.format('?'))
    fig.suptitle('Loss')
    plt.subplot(211)
    plt.semilogy(np.array(range(0,len(pureLines))), np.array(pureLines))
    plt.ylabel('Loss')
    plt.xlabel('# Iterations')
    plt.grid(True)
    fig.tight_layout()
    plt.subplot(212)
    plt.plot(np.array(range(0,means.size)), means)
    plt.ylabel('Loss')
    plt.xlabel('# Epochs')
    plt.grid(True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig('currentLoss_{}.png'.format(fileName[:fileName.index('_')]))
    plt.close(fig)
    print('\n')


def main(argv):
    #makeGraph('alexnet_Results.txt', 1125)
    #makeGraph('squeezenet_Results.txt', 1125)
    #makeGraph('resnet_Results.txt', 1125)
    #makeGraph('inception_Results.txt', 2250)
    #makeGraph('vggnet_Results.txt', 1125)
    makeGraph('densenet_Results.txt', 4500)

if __name__ == "__main__":
    main(sys.argv)
