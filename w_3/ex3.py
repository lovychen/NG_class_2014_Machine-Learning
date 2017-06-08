# -*-coding:UTF-8-*-
import scipy.io
import numpy as np

from matplotlib import use
use('TkAgg')

from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
from displayData import displayData

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                         # (note that we have mapped "0" to label 10)

print 'Loading and Visualizing Data ...'

data = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
X = data['X']
y = data['y']
print y
# 数据大小，每行400个数据，共5000 个数据
m, _ = X.shape

# 降低数据维度
y=y.flatten()
print X.shape
rand_indices = np.random.permutation(range(m))
# 显示前100 个数据
sel = X[rand_indices[0:100], :]

displayData(sel)
#
# raw_input("Program paused. Press Enter to continue...")

## ============ Part 2: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

print 'Training One-vs-All Logistic Regression...'

Lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, Lambda)

raw_input("Program paused. Press Enter to continue...")


# ================ Part 3: Predict for One-Vs-All ================
#  After ...
pred = predictOneVsAll(all_theta, X)
#
accuracy = np.mean(np.double(pred == np.squeeze(y))) * 100
print '\nTraining Set Accuracy: %f\n' % accuracy



