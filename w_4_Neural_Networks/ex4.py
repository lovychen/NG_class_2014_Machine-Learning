#!/usr/bin/env python

import scipy.io
import numpy as np

import nnCostFunction as nncf
import sigmoidGradient as sg
import randInitializeWeights as riw
#import checKNNGradients as cnng
from  scipy.optimize import minimize
#import predict as pr

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

print "start load the data"

mat = scipy.io.loadmat('ex4data1.mat')

X = mat['X']
y = mat['y']

m = X.shape[0]

y = y.flatten()

#random creat a array list

#display this picture
#rand_indices = np.random.permutation

#sel = X[rand_indices[:100],:]

#dd.displayData(sel)
mat = scipy.io.loadmat('ex4weights.mat')
Theta1 = mat["Theta1"]  # shape (25 401)

Theta2 = mat["Theta2"]   #  shape(10 26)
nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))   # shape 10285  25*401 + 10*26
lambda_reg = 0

J,ggard  = nncf.nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,\
X,y,lambda_reg)


print J
print ggard.shape
print ggard

print('Training Set Accuracy: {:f}\n(this value should be about 0.287629)'.format(J))

raw_input('Program paused. Press enter to continue.\n')

print('Checking Cost Function (w/ Regularization)...')


lambda_reg = 5

J, _ = nncf.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
 num_labels, X, y, lambda_reg)



print('Cost at parameters (loaded from ex4weights): {:f}\n(this value should be about 0.383770)'.format(J))


raw_input('Program paused. Press enter to continue.\n')



print('Training Neural Network...')


nn_params = np.zeros((10285))
maxiter = 50
lambda_reg = 0.1

myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)

results = minimize(nncf.nnCostFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)


nn_params = results["x"]

Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
(hidden_layer_size, input_layer_size + 1), order='F')

Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
(num_labels, hidden_layer_size + 1), order='F')


#pred = pr.predict(Theta1,Theta2,x)


#print('Training Set Accuracy: {:f}'.format( ( np.mean(pred == y)*100 ) ) )

