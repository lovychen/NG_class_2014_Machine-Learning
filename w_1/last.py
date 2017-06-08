# -*-coding:UTF-8-*-
# Created on 2015年10月20日
# @author: hanahimi
import numpy as np
import random
import matplotlib.pyplot as plt


def randData():
    # # 生成曲线上各个点
    # x = np.arange(-1, 1, 0.02)
    # y = [2 * a + 3 for a in x]  # 直线
    # #     y = [((a*a-1)*(a*a-1)*(a*a-1)+0.5)*np.sin(a*2) for a in x]  # 曲线
    # xa = [];
    # ya = []
    # # 对曲线上每个点进行随机偏移
    # for i in range(len(x)):
    #     d = np.float(random.randint(90, 120)) / 100
    #     ya.append(y[i] * d)
    #     xa.append(x[i] * d)
    # return xa, ya
    train_x = []
    train_y = []
    fileIn = open('ex1data1.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        # print lineArr[0]
        # print lineArr[1]
        train_x.append(float(lineArr[0]))
        train_y.append(float(lineArr[1]))
    return train_x,train_y


def hypfunc(x, A):
    # 输入：x 横坐标数值， A 多项式系数 [a0,a1,...,an-1]
    # 返回 y = hypfunc(x)
    return np.sum(A[i] * (x ** i) for i in range(len(A)))



# 使用 θ = (X.T*X + λI)^-1 * X.T * y求解直线参数
# 该函数会在X的前面添加偏移位X0 = 1
def LS_line(X, Y, lam=0.01):
    X = np.array(X)
    print X
    print type(X)
    print X.shape
    print len(X)
    X = np.vstack((np.ones((len(X),)), X))  # 往上面添加X0
    print X
    X = np.mat(X).T  # (m,n)
    Y = np.mat(Y).T  # (m,1)
    M, N = X.shape
    print X.shape
    print Y.shape
    print Y
    I = np.eye(N, N)  # 单位矩阵

    theta = ((X.T * X + lam * I) ** -1) * X.T * Y  # 核心公式
    print "----------------------------no reshape--------------------"
    print theta
    theta = np.array(np.reshape(theta, len(theta)))[0]
    print theta
    return theta  # 返回一个一维数组


# 使用随机梯度下降法求解最小二参数:
# alpha 迭代步长（固定步长），epslion 收敛标准
def LS_sgd(X, Y, alpha=0.1, epslion=0.003):
    X = [[1, xi] for xi in X]  # 补上偏移量x0
    N = len(X[0])  # X的维度
    M = len(X)  # 样本个数
    theta = np.zeros((N,))  # 参数初始值
    last_theta = np.zeros(theta.shape)

    times = 300
    while times > 0:
        times -= 1
        for i in range(M):
            last_theta[:] = theta[:]
            for j in range(N):
                theta[j] -= alpha * (np.dot(theta, X[i]) - Y[i]) * X[i][j]
        print "theta:",theta
        # if np.sum((theta - last_theta) ** 2) <= epslion:  # 当前后参数的变化小于一定程度时可以终止迭代
        #     break
    return theta


# 根据输入值：X向量，即拟合阶数，计算对应的范德蒙矩阵
def vandermonde_matrix(X, Y, order=1):
    # 根据数据点构造X，Y的 范德蒙德矩阵
    m = len(Y)
    matX = np.array([[np.sum([X[i] ** (k2 + k1) for i in range(m)])
                      for k2 in range(order + 1)] for k1 in range(order + 1)])
    matY = np.array([np.sum([(X[i] ** k) * Y[i] for i in range(m)])
                     for k in range(order + 1)])
    print "-------+++++++++++++-------------------"
    print matX
    print matY
    theta = np.linalg.solve(matX, matY)
    print theta
    return theta


if __name__ == "__main__":
    pass
    X, Y = randData()
    print X
    print Y
    theta_1 = vandermonde_matrix(X, Y, order=1)
    theta_2 = LS_sgd(X, Y)
    theta_3 = LS_line(X,Y)
    print "last:", theta_1
    print "sgd:", theta_2
    print "line:",theta_3

    # 画出数据点与拟合曲线
    plt.figure()
    plt.plot(X, Y, linestyle='', marker='.')
    yhyp = [hypfunc(X[i], theta_3) for i in range(len(X))]
    plt.plot(X, yhyp, linestyle='-')
    plt.show()