def sigmoid(z):
    from scipy.special import expit
    import numpy as np
    
    g = np.zeros(z.shape)
   
    g = expit(z)

    return  g
