import numpy as np
import scipy.linalg as linalg

def pca(X):
    m,n = X.shape

    U = np.zeros(n)
    S = np.zeros(n)

    sigma = (1.0/m)*(X.T).dot(X)
    print sigma.shape

    U,S,Vh = linalg.svd(sigma)
    print S.shape
    S = linalg.diagsvd(S,len(S),len(S))

    return U,S
