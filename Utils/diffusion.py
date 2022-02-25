from scipy.sparse.csgraph import shortest_path, laplacian
from scipy.linalg import expm
import numpy as np


def diffuse(Y, tau):

    Cs, Ls = Y[0], Y[1]
    Ls_diff = []
    Cs_dist = []

    for i in range(len(Cs)):
        C, L = Cs[i], Ls[i]
        C = np.ascontiguousarray(C)
        A = np.zeros(C.shape)
        A[C > 1 / 2] = 1
        A_reg = (A + 1e-2 * np.ones(A.shape))
        Cs_dist.append(A_reg)
        C = A
        Lap = laplacian(C, normed=True)
        A = expm(- tau * Lap)
        # L = L + 1e-4 * np.ones(L.shape)
        L_diff = A.dot(L)
        Ls_diff.append(L_diff)

    Y[0] = Cs_dist
    Y[1] = Ls_diff

    return Y
