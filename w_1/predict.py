# -*-coding:UTF-8-*-
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import time
import last

# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# train a logistic regression model using some optional optimize algorithm
# input: train_x is a mat datatype, each row stands for one sample
#		 train_y is mat datatype too, each row is the corresponding label
#		 opts is optimize option include step and maximum number of iterations
def trainLogRegres(train_x, train_y, opts):
    # calculate training time
    startTime = time.time()
    numSamples, numFeatures = shape(train_x)
    print numSamples
    print numFeatures

    alpha = opts['alpha'];
    maxIter = opts['maxIter']
    weights = zeros((numFeatures, 1))
    print weights.shape
    # optimize through gradient descent algorilthm

    for k in range(maxIter):
        # 批量梯度下降算法
        if opts['optimizeType'] == 'gradDescent':  # gradient descent algorilthm
            output = train_x * weights
            error = (output - train_y)
            print error.shape
            print train_x.shape
            weights = weights - alpha * train_x.transpose()*error/numSamples

        #单一梯度下降算法
        elif opts['optimizeType'] == 'stocGradDescent':  # stochastic gradient descent
            for i in range(numSamples):
                output = train_x[i, :] * weights
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose() * error

        #随机梯度下降算法
        elif opts['optimizeType'] == 'smoothStocGradDescent':  # smooth stochastic gradient descent
            # randomly select samples to optimize for reducing cycle fluctuations
            dataIndex = range(numSamples)
            for i in range(numSamples):
                # alpha = 4.0 / (1.0 + k + i) + 0.01
                alpha = 0.0001
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = train_x[randIndex, :] * weights
                error = train_y[randIndex, 0] - output
                print weights
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del (dataIndex[randIndex])  # during one interation, delete the optimized sample
        #最小二乘法直接求值,一步求解
        elif opts['optimizeType'] == "LeastSquareMethod":
            lam = 0.001
            X = np.array(train_x[:,0])
            X = X.ravel()
            X = np.vstack((X,np.ones(len(X))))  # 添加一个维度全部为1
            print X
            X = np.mat(X).T  # (m,n)
            Y = np.mat(train_y)  # (m,1)
            M, N = X.shape
            I = np.eye(N, N)  # 单位矩阵

            weights = ((X.T * X + lam * I) ** -1) * X.T * Y  # 核心公式
            # weights = np.array(np.reshape(weights, len(weights)))[0]
            # weights = weights.reshape(-1,1)
            break
        elif opts["optimizeType"] == "Vandermonde":
            # 根据数据点构造X，Y的 范德蒙德矩阵
            order = 1
            m = len(train_y)
            X = np.array(train_x[:, 0])
            X = X.ravel()
            Y = train_y
            matX = np.array([[np.sum([X[i] ** (k2 + k1) for i in range(m)])
                              for k2 in range(order + 1)] for k1 in range(order + 1)])
            matY = np.array([np.sum([(X[i] ** k) * Y[i] for i in range(m)])
                             for k in range(order + 1)])
            print matX
            print matY
            theta = np.linalg.solve(matX, matY)
            aa = [[theta[1]],[theta[0]]]
            weights = np.array((aa))
        else:
            raise NameError('Not support optimize method type!')


    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return weights


# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = shape(test_x)
    matchCount = 0
    for i in xrange(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# show your trained logistic regression model only available with 2-D data
def showLogRegres(weights, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = shape(train_x)
    if numFeatures != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        plt.plot(train_x[i, 0],train_y[i,0], 'or')

    # draw the classify line
    min_x = min(train_x[:, 0])[0, 0]
    max_x = max(train_x[:, 0])[0, 0]

    print min_x
    print max_x
    # weights = weights.getA()  # convert mat to array
    print weights
    # weights = [[1],[01]]
    weights = np.array((weights))
    y_min_x = float(weights[1] + weights[0] * min_x)
    y_max_x = float(weights[1] + weights[0] * max_x)
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()

def loadData():
    train_x = []
    train_y = []
    fileIn = open('ex1data2.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        train_x.append([float(lineArr[0]),1.0])
        train_y.append(float(lineArr[1]))
    return mat(train_x), mat(train_y).transpose()


# step 1: load data
print "step 1: load data..."
train_x, train_y = loadData()
test_x = train_x; test_y = train_y

# step 2: training...
print "step 2: training..."
opts = {'alpha': 0.001, 'maxIter': 60, 'optimizeType': 'Vandermonde'}
optimalWeights = trainLogRegres(train_x, train_y, opts)
print optimalWeights

# step 3: testing
print "step 3: testing..."
accuracy = testLogRegres(optimalWeights, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
showLogRegres(optimalWeights, train_x, train_y)