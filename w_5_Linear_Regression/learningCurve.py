import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf


def learningCurve(X,y,Xval,yval,lambda_val):
    m = len(X)
    error_train = np.zeros((m,1))
    error_val = np.zeros((m,1))
    for i in xrange(1,m+1):
        X_train = X[:i]
        y_train = y[:i]

        theta = tlr.trainLinearReg(X_train,y_train,1)

        error_train[i-1] = lrcf.linearRegCostFunction(X_train,y_train,theta,0)
        error_val[i-1] = lrcf.linearRegCostFunction(Xval,yval,theta,0)

    return error_train,error_val