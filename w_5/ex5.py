
import os
import sys
import scipy.io
import numpy as np

import linearRegCostFunction as lrcf

import trainLinearReg as tlr

import polyFeatures as pf

import learningCurve as lc

import matplotlib.pyplot as plt


print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('ex5data1.mat')

X = mat['X']
y = mat['y']
Xval = mat['Xval']
yval = mat['yval']
Xtest = mat['Xtest']
ytest = mat['ytest']

print Xval
m = X.shape[0]

plt.plot(X,y,'rx',markersize=10,linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
# plt.show(block=False)
plt.show()
raw_input('Program paused. Press enter to continue.\n')

theta = np.array([[1], [1]])
X_padded = np.column_stack((np.ones((m,1)), X))
J = lrcf.linearRegCostFunction(X_padded,y,theta,1)

print('Cost at theta = [1 ; 1]: {:f}\n(this value should be about 303.993192)\n'.format(J))

raw_input('Program paused. Press enter to continue.\n')

theta = np.array([[1] , [1]])

J, grad = lrcf.linearRegCostFunction(X_padded, y, theta, 1, True)

print('Gradient at theta = [1 ; 1]:  [{:f}; {:f}] \n(this value should be about [-15.303016; 598.250744])'.format(grad[0], grad[1]))

raw_input('Program paused. Press enter to continue.\n')

print "Part 4: Train Linear Regression "

lambda_val = 0
theta = tlr.trainLinearReg(X_padded, y, lambda_val)

#display
plt.close()
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.hold(True)
plt.plot(X, np.dot(np.column_stack((np.ones((m,1)), X)), theta), '--', linewidth=2)
plt.show()
raw_input('Program paused. Press enter to continue.\n')
plt.hold(False)

## =========== Part 5: Learning Curve for Linear Regression =============

lambda_val = 0
error_train, error_val = lc.learningCurve(np.column_stack((np.ones((m,1)), X)), y, np.column_stack((np.ones((Xval.shape[0], 1)), Xval)), yval, lambda_val)



# resets plot
plt.close()

p1, p2 = plt.plot(range(m), error_train, range(m), error_val)
plt.title('Learning curve for linear regression')
plt.legend((p1, p2), ('Train', 'Cross Validation'), numpoints=1, handlelength=0.5)
plt.xlabel('Number of training examples')
plt.ylabel('Error')



plt.show()
plt.axis([0, 13, 0, 150])
raw_input('Program paused. Press enter to continue.\n')

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in xrange(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))

raw_input('Program paused. Press enter to continue.\n')

## =========== Part 6: Feature Mapping for Polynomial Regression =============

p = 8;
X_poly = pf.ployFeature(X,p)

X_poly, mu, sigma = fn.featureNormalize(X_poly)

X_ploy = np.column_stack((np.ones((m,1)),X_poly))
