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

def get_statistics(A_train, L_train, A_s_train, L_s_train,
                   A_weak, L_weak, A_s_weak, L_s_weak,
                   A_test, L_test, A_s_test, L_s_test):
    print("L2")
    print("Train L2 {:.5f}".format((np.linalg.norm(A_train - A_s_train, axis=1)**2).mean()))
    print("Weak L2 {:.5f}".format((np.linalg.norm(A_weak - A_s_weak, axis=1)**2).mean()))
    print("Test L2 {:.5f}".format((np.linalg.norm(A_test - A_s_test, axis=1)**2).mean()), end="\n\n")

    print("MWD")
    print("Train L2 {:.5f}".format(get_Wasserstain(A_train, L_train, A_s_train, L_s_train)))
    print("Weak L2 {:.5f}".format(get_Wasserstain(A_weak, L_weak, A_s_weak, L_s_weak)))
    print("Test L2 {:.5f}".format(get_Wasserstain(A_test, L_test, A_s_test, L_s_test)))
    