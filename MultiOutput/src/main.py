import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

def get_W_k_means(X, n_clusters = 4, n_models = 4, normalize_W=True, verbose=False):
    X_size = X.shape[0]
    idxes = np.arange(X_size)
    np.random.shuffle(idxes)
    batch_size = math.ceil(X_size/n_models)
    idxes_batches = []
    for i in range(n_models):
        idxes_batches.append(idxes[i*batch_size:(i+1)*batch_size])

    models = [KMeans(n_clusters=n_clusters) for i in range(n_models)]
    W = np.zeros((X.shape[0], X.shape[0]))
    
    if verbose:
        print("Fitting {} KMeans models: ".format(n_models), end="")
        
    for i in range(n_models):
        if verbose:
            print(i+1, end=" ")
        models[i].fit(X[idxes_batches[i]])
        pred = models[i].predict(X)
        W += ((pred[:, None] - pred[None,: ]) == 0).astype(int)
        
    W /= n_models
    if normalize_W:
        W = (W/(W.sum(1)[:, None]))
        
    return W

def get_W_gaussian(X, Y=None, gamma=1, normalize_W=False):
    W = rbf_kernel(X, X, gamma)
    if normalize_W:
        W = (W/W.sum(1)[:, None])
    return W

def solve(W, B, Y, S, gamma=0.01): 
    D = np.diag(np.sum(W, axis=1))
    coeff = np.linalg.inv(B + 2*gamma*(D - W))
    
    return coeff@Y, coeff@S