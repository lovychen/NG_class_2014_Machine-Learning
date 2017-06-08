# -*-coding:UTF-8-*-
import numpy as np
import lrCostFunction as lrcf
from scipy.optimize import minimize
# from lrCostFunction import lrgradientReg
from lrCostFunction import lrCostFunction

from scipy import optimize

def oneVsAll(X,y,num_labels,lambda_reg):
    m,n = X.shape
    print m , n   #(5000L,400L)
    all_theta = np.zeros((num_labels, n + 1))
    X = np.column_stack((np.ones((m, 1)), X))

    # initial_theta = np.zeros((X.shape[1], 1)).reshape(-1)  # 400 * 1   #初始化 要得到的数据
    #
    # Theta = np.zeros((num_labels, X.shape[1]))  # 10 * 5000  这是分类结果
    for c in xrange(num_labels):

        initial_theta = np.zeros((n+1,1))
        print("Training {:d} out of {:d} categories...".format(c + 1, num_labels))
        # print '----------------gg-------------------'
        # gg = 0
        # for each in y:
        #     if each == 10:
        #         gg = gg+ 1
        # print gg
        # return
        myargs = (X,(y%10==c).astype(int),lambda_reg, True)
        print type(X)
        print type(initial_theta)
        print X.shape
        print initial_theta.shape
        theta = minimize(lrcf.lrCostFunction,x0=initial_theta,args=myargs,options={'disp':True,'maxiter':13},method='Newton-CG',jac=True)
        print theta
        all_theta[c,:] = theta['x']
    return all_theta
