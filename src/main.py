import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

def get_W_k_means(X, n_clusters = 4, n_models = 4, verbose=False):
    models = [KMeans(n_clusters=n_clusters) for i in range(n_models)]
    W = np.zeros((X.shape[0], X.shape[0]))
    
    if verbose:
        print("Fitting {} KMeans models: ".format(n_models), end="")
        
    for i in range(n_models):
        if verbose:
            print(i+1, end=" ")
        #models[i].fit(X)
        pred = models[i].fit_predict(X)
        W += ((pred[:, None] - pred[None,: ]) == 0).astype(int)
        
    W /= n_models
    return W

def get_W_gaussian(X, Y=None, gamma=1):
    W = rbf_kernel(X, X, gamma)
    return W

def solve(W, B, Y, S, gamma=0.01):

    D = np.diag(np.sum(W, axis=1))
    coeff = np.linalg.inv(B + 2*gamma*(D - W))
    
    return coeff@Y, coeff@S

def solve_sparse(D, C_1, C_2, B, Y, L, gamma=0.01):
    G = np.diag(1/np.diag(B + 2*gamma*D))
    U = -2*gamma*C_1
    V = C_2
    GU = G@U
    VGU = V@GU
    M = G - GU@np.linalg.inv(np.eye(VGU.shape[0]) + VGU)@V@G
    return M@Y, M@L

class Nystrom():
    
    def __init__(self, rank):
        self.r = rank
        
    def decompose(self, M):
        C = M[:, :self.r]
        W = C[:self.r, :self.r]
        return C, np.linalg.pinv(W)@C.T
        
    def __call__(self, M):
        return self.decompose(M)
    