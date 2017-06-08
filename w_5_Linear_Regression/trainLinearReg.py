import numpy as np
import linearRegCostFunction as lrcf
from scipy.optimize import minimize
def trainLinearReg(X,y,lambda_val):
    initial_theta = np.zeros((X.shape[1],1))

    def costFunc(theta):
        return lrcf.linearRegCostFunction(X,y,theta,lambda_val,True)

    maxiter = 200

    results = minimize(costFunc,x0 = initial_theta,options = {'disp':True,'maxiter':maxiter} , method="L-BFGS-B", jac=True)

    theta = results['x']

    return theta

