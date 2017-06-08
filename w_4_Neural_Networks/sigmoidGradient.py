import numpy as np

def sigmoidGradient(z):


    g =  1.0 / (1.0 + np.exp(-z))
    
    g = g*(1-g)
    
    return g
