import time
from numpy import *

def trainSVM(train_x,train_y,C,tiler,maxIter,lernelOption = ('rbf',1.0)):
    startTime = time.time()

    svm = SVMStruct