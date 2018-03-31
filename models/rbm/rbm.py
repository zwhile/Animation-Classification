import os
import sys
import matplotlib.image as mpimg
import cv2
import numpy as np
from skimage import io
from PIL import Image
import timeit
import glob
from tqdm import tqdm
import random

#from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import argparse
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()


def getSet(inPath):
    inPath = inPath[inPath.index('/T')+1:]
    inPath = inPath[:inPath.index('g/')+1]
    return inPath


def getLabel(inPath):
    inPath = inPath[inPath.index('g/')+2:]
    #print('c: -{}-'.format(inPath))
    #print(inPath.index('/'))
    inPath = inPath[:inPath.index('/')]
    return inPath


def loadImages(whichSet):
    X = []
    y = []
    if whichSet == 'train':
        picPaths = glob.glob("/workspace/shared/biometrics-project/Data/Training/*/*.png")
        random.shuffle(picPaths)
        #print('num: {}'.format(len(picPaths)))
        for i in tqdm(picPaths):
            picVector = cv2.resize(cv2.imread(i), (28,28)).astype(np.float32)/255.0
            #if picVector.shape != (224, 224, 3):
                #print('{}: {}'.format(getLabel(i), picVector.shape))
            X.append(np.ndarray.flatten(picVector))
            #print(np.ndarray.flatten(picVector).shape)
            y.append(getLabel(i))
        npX = np.asarray(X)
        print('npX.shape: {}'.format(npX.shape))
        #with h5py.File('data.h5', 'w') as hf:
            #hf.create_dataset('train',  data=npX)

    elif whichSet == 'test':
        picPaths = glob.glob("/workspace/shared/biometrics-project/Data/Testing/*/*.png")
        random.shuffle(picPaths)
        #print('num: {}'.format(len(picPaths)))
        for i in tqdm(picPaths):
            #print(i)
            picVector = cv2.resize(cv2.imread(i), (28,28)).astype(np.float32)/255.0
            #print(picVector.shape)
            #if picVector.shape != (224, 224, 3):
                #print('{}: {}'.format(getLabel(i), picVector.shape))
                #print(picVector.shape)
            X.append(np.ndarray.flatten(picVector))
            #print(np.ndarray.flatten(picVector).shape)
            y.append(getLabel(i))
        npX = np.asarray(X)
        print('npX.shape: {}'.format(npX.shape))
        #with h5py.File('data.h5', 'w') as hf:
            #hf.create_dataset('test',  data=npX)

    return(npX, y)


def main(argv):

    orig_stdout = sys.stdout
    with open('out.txt', 'w') as f:
        sys.stdout = f

        print('\n')

        search = True

        (trainX, trainY) = loadImages('train')
        (testX, testY) = loadImages('test')

        if search:
            print("SEARCHING LOGISTIC REGRESSION")
            params = {"C": [1.0, 10.0, 100.0]}
            start = time.time()
            gs = GridSearchCV(LogisticRegression(), params, n_jobs = -1, verbose = 1)
            gs.fit(trainX, trainY)
            print('best score: {}'.format(gs.best_score_))
            bestParams = gs.best_estimator_.get_params()
            for p in sorted(params.keys()):
                print('{}: {}'.format(p, bestParams[p]))
            rbm = BernoulliRBM()
            logistic = LogisticRegression()
            classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
            print('\n\nSearching RBM + Logistic Regression')
            params = {
                "rbm__learning_rate": [0.1, 0.01, 0.001],
                "rbm__n_iter": [20, 40, 80],
                "rbm__n_components": [50, 100, 200],
                "logistic__C": [1.0, 10.0, 100.0]}
            gs = GridSearchCV(classifier, params, n_jobs = -1, verbose = 1)
            gs.fit(trainX, trainY)
            print('best score: {}'.format(gs.best_score_))
            bestParams = gs.best_estimator_.get_params()
            for p in sorted(params.keys()):
                print('{}: {}'.format(p, bestParams[p]))

        else:
            logistic = LogisticRegression(C = 1.0)
            logistic.fit(trainX, trainY)
            print("LOGISTIC REGRESSION ON ORIGINAL DATASET")
            print(classification_report(testY, logistic.predict(testX)))

            rbm = BernoulliRBM(n_components = 200, n_iter = 40,
                learning_rate = 0.01,  verbose = True)
            logistic = LogisticRegression(C = 1.0)

            classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
            classifier.fit(trainX, trainY)
            print("RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET")
            print(classification_report(testY, classifier.predict(testX)))

                # nudge the dataset and then re-evaluate
                #print "RBM + LOGISTIC REGRESSION ON NUDGED DATASET"
                #(testX, testY) = nudge(testX, testY)
                #print classification_report(testY, classifier.predict(testX))

        print('\n')
        sys.stdout = orig_stdout
    #trainImg = glob.glob("/workspace/shared/biometrics-project/Data/Training/*/*.png")
    #testImg = glob.glob("/workspace/shared/biometrics-project/Data/Testing/*/*.png")
    #allImg = glob.glob("/workspace/shared/biometrics-project/Data/*/*/*.png")
    #print('train: {}'.format(len(trainImg)))
    #print('test: {}'.format(len(testImg)))
    #print('all: {}'.format(len(allImg)))

    #classes = os.listdir(os.path.join(os.getcwd(), 'Data', 'Training'))
    #print(classes)

    #random.shuffle(allImg)
    #print('')

    #for i in allImg[0:15]:
        #foo = i
        #orig = foo
        #print(orig)
        #print(getSet(foo))
        #print(getLabel(foo))
        #print('')
        #print(foo)
        #print(foo[foo.index('/workspace/shared/biometrics-project/Data/'):])
        #foo = foo[foo.index('/T')+1:]
        #print(foo)
        #foo = foo[foo.index('g/')+2:]
        #foo = foo[:foo.index('/')]
        #print(foo)
        #if foo not in classes:
            #print('issue: {}'.format(foo))
        #print('-{}-'.format(foo))
        #if foo != 'Training' and foo != 'Testing':
            #print('issue: {} from {}'.format(foo, orig))

    #/workspace/shared/biometrics-project/Data/Training/Fosters/1618.png
    #for i in allImg[0:10]:
        #print(i)

    #allPics = []
    #for i in tqdm(sorted(allImg)):
    #    allPics.append(cv2.imread(i).astype(np.float32)/255.0)


    '''
    times = range(1000)

    # matplotlib
    start_time = timeit.default_timer()
    for t in times:
        img = mpimg.imread('img1.png')
    print("mpimg.imread(): ", timeit.default_timer() - start_time, "s")

    # OpenCV
    start_time = timeit.default_timer()
    for t in times:
        img = cv2.cvtColor(
            cv2.imread('img1.png'), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    print("cv2.imread(): ", timeit.default_timer() - start_time, "s")

    # scikit-image
    start_time = timeit.default_timer()
    for t in times:
        img = io.imread('img1.png').astype(np.float32)/255.0
    print("io.imread(): ", timeit.default_timer() - start_time, "s")

    # PIL
    start_time = timeit.default_timer()
    for t in times:
        img = np.asarray(Image.open('img1.png')).astype(np.float32)/255.0
    print("Image.open(): ", timeit.default_timer() - start_time, "s")
    '''


if __name__ == "__main__":
    main(sys.argv)
