from numpy import *


import matplotlib.pyplot as plt

import showdata as sd

import train_svm as ts


def loadData():
    train_x = []
    train_y = []
    fileIn = open('ex2data1.txt')
    # fileIn = open('testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        train_x.append([float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()
dataSet,labels = loadData()

train_x = dataSet[:80,:]
train_y = labels[:80,:]
print train_x.shape
test_x = dataSet[80:,:]
test_y = labels[80:,:]
print test_y.shape

# display the data
print "show the data......"
sd.showdata(train_x,train_y)

print "setp 2:start training......."

C  =0.6
toler = 0.001
maxIter = 50

svmClassifier = ts.trainSVM(train_x,train_y,C,toler,maxIter,kernelOption = ('linear',0))