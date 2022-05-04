import math
import numpy as np

def get_Wasserstain(A, L, A_star, L_star):
    get_S = lambda x: x@x.T
    L_m = np.stack([tril_inv(L[i]) for i in range(L.shape[0])])
    L_star_m = np.stack([tril_inv(L_star[i]) for i in range(L_star.shape[0])])
    return np.mean(
        np.linalg.norm(A-A_star, ord=2, axis=1)**2+
        np.linalg.norm(L_m-L_star_m, ord="fro", axis=(1, 2))**2
    )

def get_shape(n):
    return int((-1+math.sqrt(1 + 8*n))/2)

def tril_inv(L):
    shape = get_shape(len(L))
    _S = np.zeros((shape, shape))
    _S[np.tril_indices(shape)] = L
    return _S