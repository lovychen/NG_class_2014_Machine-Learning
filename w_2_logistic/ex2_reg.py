# -*-coding:UTF-8-*-
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
import costFunctionReg as cfr
import time
import os
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
# load the dataset
data = np.loadtxt('ex2data2.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]
pos = where(y == 1)
neg = where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature1/Exam 1 score')
ylabel('Feature2/Exam 2 score')
legend(['Fail', 'Pass'])
show()

def sigmoid(X):
    '''''Compute sigmoid function '''
    den = 1.0 + e ** (-1.0 * X)
    gz = 1.0 / den
    return gz


def compute_cost(theta, X, y):
    '''''computes cost given predicted and actual values'''
    m = X.shape[0]  # number of training examples
    theta = reshape(theta, (len(theta), 1))

    J = (1. / m) * (-transpose(y).dot(log(sigmoid(X.dot(theta)))) - transpose(1 - y).dot(log(1 - sigmoid(X.dot(theta)))))

    grad = transpose((1. / m) * transpose(sigmoid(X.dot(theta)) - y).dot(X))
    # optimize.fmin expects a single value, so cannot return grad
    return J[0][0]  # ,grad


def compute_grad(theta, X, y):
    '''''compute gradient'''
    theta.shape = (1, 3)
    grad = zeros(3)
    h = sigmoid(X.dot(theta.T))
    delta = h - y
    l = grad.size
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / m) * sumdelta * -1
    theta.shape = (3,)
    return grad

def map_feature(x1, x2):
    '''''
    Maps the two input features to polonomial features.
    Returns a new feature array with more features of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    '''
    x1.shape =(x1.size,1)
    x2.shape =(x2.size,1)
    degree =6
    mapped_fea = ones(shape=(x1[:,0].size,1))
    m, n = mapped_fea.shape
    for i in range(1, degree +1):
        for j in range(i +1):
            r =(x1 **(i - j))*(x2 ** j)
            mapped_fea = append(mapped_fea, r, axis=1)
    return mapped_fea
mapped_fea = map_feature(X[:,0], X[:,1])
def cost_function_reg(theta, X, y, l):
    '''''Compute the cost and partial derivatives as grads
    '''
    h = sigmoid(X.dot(theta))
    thetaR = theta[1:,0]
    J =(1.0/ m)*((-y.T.dot(log(h)))-((1- y.T).dot(log(1.0- h)))) +(l /(2.0* m))*(thetaR.T.dot(thetaR))
    delta = h - y
    sum_delta = delta.T.dot(X[:,1])
    grad1 =(1.0/ m)* sum_delta
    XR = X[:,1:X.shape[1]]
    sum_delta = delta.T.dot(XR)
    grad =(1.0/ m)*(sum_delta + l * thetaR)
    out = zeros(shape=(grad.shape[0], grad.shape[1]+1))
    out[:,0]= grad1
    out[:,1:]= grad
    return J.flatten(), out.T.flatten()
print "-------------get train shape---------------"
#初始化数据
m, n = X.shape

y.shape =(m,1)
it = map_feature(X[:,0], X[:,1])
initial_theta = np.zeros(shape=(it.shape[1],1))   #theta de 维度（28,1），全部为0
lamb = 1
# print "----------------m1---------------------"
#Use regularization and set parameter lambda to 1
# Compute and display initial cost and gradient for regularized logistic
# regression
# cost, grad = cost_function_reg(initial_theta, it, y, lamb)

# print "---------------cose    grad----------"
#
# print cost,grad
# def decorated_cost(theta):
#     return cost_function_reg(theta, X, y, lamb)
# # print fmin_bfgs(decorated_cost, initial_theta, maxfun=500)
# #
# #Plot Boundary
# print "---------------------------m1 end---------------"

# 直接调用之前的方法
print "------------------------m2-------------------------------"
y = y.ravel()
myargs = (it,y,lamb)
print initial_theta.shape
print y.shape
# cost = cfr.costFunctionReg(initial_theta, it, y, l)
theta_ = fmin_bfgs(cfr.costFunctionReg, x0 = initial_theta, args=myargs)
theta =  theta_
print "------------------------m2 end-----------------------------"

print theta
u = linspace(-1,1.5,50)
v = linspace(-1,1.5,50)
z = zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j]=(map_feature(array(u[i]), array(v[j])).dot(array(theta)))
z = z.T

plt.contour(u, v, z)
plt.title('lambda = %f'% lamb)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
legend(['y = 1','y = 0','Decision boundary'])
show()
# def predict(theta, X):
#     '''''Predict whether the label
#     is 0 or 1 using learned logistic
#     regression parameters '''
#     m, n = X.shape
#     p = zeros(shape=(m,1))
#     h = sigmoid(X.dot(theta.T))
#     for it in range(0, h.shape[0]):
#         if h[it]>0.5:
#             p[it,0]=1
#         else:
#             p[it,0]=0
#     return p
# #% Compute accuracy on our training set
# p = predict(array(theta), it)
# print'Train Accuracy: %f'%((y[where(p == y)].size / float(y.size))*100.0)