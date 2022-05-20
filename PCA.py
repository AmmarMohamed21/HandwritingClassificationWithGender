import pandas as pd
import numpy as np
from scipy.optimize import minimize ## if not installed, use conda install -c anaconda scipy
import matplotlib.pyplot as plt
import scipy.io 

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X,axis=0)
    normalized_X = (X-mu)/sigma

    return (normalized_X, mu, sigma)

def pca(X):
    u,s,vh=np.linalg.svd(X)
    return vh.T, s / (X.shape[0] -1)

def projectData(X, U, K):
    Z = np.matmul(X, U[:,:K])
    return Z

def recoverData(Z, U, K):
    X_rec = np.matmul(Z, U[:,:K].T)
    return X_rec