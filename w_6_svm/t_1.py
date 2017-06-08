#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import t_1_svm as SVM
import showdata
################## test w_6_svm #####################
## step 1: load data
print "step 1: load data..."
dataSet = []
labels = []
fileIn = open('ex2data2.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split(',')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])
    labels.append(float(lineArr[2]))


dataSet = mat(dataSet)
labels = mat(labels).T
print labels
for i,each in enumerate(labels):
    if labels[i,0] == 0.0:
        labels[i,0] = -1.0
train_x = dataSet[0:81, :]
train_y = labels[0:81, :]

test_x = dataSet[80:101, :]
test_y = labels[80:101, :]
# showdata.showdata(dataSet,labels)

#---------------------------
train_x = dataSet
train_y = labels

t1 = [[-0.25,0.25],[0.25,0.25],[0.25,0.00],[-0.75,-0.75],[-0.25,-0.75],[1.,1.],[1.,-0.75],[-0.75,1.]]
t2 = [[1],[1],[1],[-1],[-1],[-1],[-1],[-1]]
## step 2: training...
print "step 2: training..."
C = 0.6
toler = 0.001
maxIter = 50
test_x = mat(t1)
test_y = mat(t2)
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('rbf', 0))


#
# ## step 3: testing
# print "step 3: testing..."
# accuracy = SVM.testSVM(svmClassifier, test_x, test_y)
#
# ## step 4: show the result
# print "step 4: show the result..."
# print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
# # SVM.showSVM(svmClassifier)
