# -*- coding:utf-8 -*-
import numpy as np

from numpy import *
import matplotlib.pyplot as plt

def sigmoid(x):
    output = 1 / (1+np.exp(-x))
    return output
def sod(output):  # 激活函数的导数
    return output * (1 - output)

def loadData():
    train_x = []
    train_y = []
    fileIn = open('D:\code\git\ml_learn\Ng_class\w_2\ex2data1.txt')
    # fileIn = open('testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        train_x.append([float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return np.array(train_x), np.array(train_y)


#w = T.dscalar('w')
#b = T.dscalar('b')
#x = T.dscalar('x')
#y = T.dscalar('y')

w = 0.001
b = 0.001
#产生数据
x_data,label = loadData()
#cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
# cluster3 = np.random.uniform(3.0, 6, (2, 100))

# label = []
# for i in range(0,100):
#     label.append(1)
#
# for i in range(0,100):
    # label.append(-1)

# x_data=np.hstack((cluster1, cluster3)).T

alpha = 0.0001
def net_input(m,n):
        return m*w+b-n

def predict(m, n):
    return net_input(m, n)

def showLogRegres(weights, train_x, train_y):
    print train_x
    print train_y
    print weights
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = shape(train_x)
    if numFeatures != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        if int(train_y[i]) == 0:
            plt.plot(train_x[i, 0], train_x[i, 1], 'or')
        elif int(train_y[i]) == 1:
            plt.plot(train_x[i, 0], train_x[i, 1], 'ob')

    # draw the classify line
    min_x = min(train_x[:, 0])
    max_x = max(train_x[:, 0])
    # weights = weights.getA()  # convert mat to array
    y_min_x = float(weights[0] +weights[1] * min_x)
    y_max_x = float(weights[0] +weights[1] * max_x)
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()

errors = []
for j in range(10):
    error = 0
    for i, each in enumerate(x_data):
        print '-----------aaaa--------bbbbb-----------------------'
        print each
        a = each[0]
        b = each[1]
        print a
        print b
        update = alpha * (label[i] - predict(a, b))
        print update
        print "update:", update
        w += update * a
        b += update
        print "w,b:", w, b
        print "-----------------++++---------------------"
        error += int(update != 0.0)
        # ann = get_ans(a,b)
    errors.append(error)
aa = [b,w]
print aa
# aa = [90.11,-0.5]
# weights = np.array((aa))
showLogRegres(aa,x_data,label)
# while 1:
#     aa = raw_input()
#     bb = raw_input()
#     a_int = float(aa)
#     b_int = float(bb)
#     print a_int
#     print b_int
#     ans = a_int * w + b - b_int
#     if ans > 0:
#         print "Hight"
#     else:
#         print "Low"
