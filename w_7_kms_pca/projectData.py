import numpy as np

# just need the haed of K
def projectData(X, U, K):
    Z = np.zeros((X.shape[0], K))
    U_reduce = U[:,:K]
    Z = X.dot(U_reduce)
    return Z