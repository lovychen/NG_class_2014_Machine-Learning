import numpy as np

def ployFeatures(X,p):
    X_poly = X

    if p>2:
        for k in xrange(1,p):
            X_plot = np.column_stack( (X_poly,np.power(X,k+1) ) )

    return X_plot