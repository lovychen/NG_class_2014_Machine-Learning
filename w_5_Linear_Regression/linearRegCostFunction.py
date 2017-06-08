import os
import sys

import numpy as np

def linearRegCostFunction(X,y,theta,lambda_val,return_grad=False):
    m = len(y)
    theta = np.reshape(theta, (-1,y.shape[1]))
    J = 0

    J = ( 1./(2*m) ) * np.power( (np.dot(X,theta) - y) , 2).sum() + (float(lambda_val) /(2*m) ) * np.power(theta[1:theta.shape[0]],2).sum()

    grad = (1./m) *np.dot(X.T,np.dot(X,theta)-y) + (float (lambda_val)/m)*theta

    grad_no_regularization = (1./m) * np.dot(X.T, np.dot(X,theta)-y)   # when j == 0

    grad[0] = grad_no_regularization[0]

    if return_grad == True:
         return J,grad.flatten()
    elif return_grad == False:
         return J
