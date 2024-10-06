import numpy as np

def tan_inv(X,Y):
    return np.arctan2(Y, X)

def distance(X, Y):
    return np.sqrt(np.square(X) + np.square(Y))

def ssa(angle):
    return (angle + np.pi)%(2*np.pi) - np.pi