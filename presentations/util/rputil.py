import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import scipy.linalg

def svd_rank_k(A, k,full_m=True):
    """Calculates a deterministic rank-k approximation of a singular value decomposition of the input matrix."""
    U, sigma, Vh = np.linalg.svd(A, full_matrices=full_m)
    Sigma = np.diag(sigma[:k])
    return U[:,:k] @ Sigma @ Vh[:k]

def random_svd_rank_k(A, k, power=1, full_m=True):
    """Calculates a randomized rank-k approximation of a singular value decomposition of the input matrix."""
    omega = random.randn(A.shape[1], k)
    Y = A @ omega
    if power != 0:
        Y = np.linalg.matrix_power(A @ A.T, power) @ Y
    Q, R = np.linalg.qr(Y)
    B = Q.T @ A
    U_tilde, sigma, Vh = np.linalg.svd(B, full_matrices=full_m)
    U = Q @ U_tilde
    Sigma = np.diag(sigma)
    return U @ Sigma @ Vh[:k]

def id_rank_k(A, k, full_m='full'):
    """Calculates a deterministic rank-k approximation of an interpolative decomposition of the input matrix."""
    Q, R, P = scipy.linalg.qr(A, pivoting=True, mode=full_m)
    Q = Q[:,:k]
    return Q @ Q.T @ A

def random_id_rank_k(A, k, oversampling=10, full_m='full'):
    """Calculates a randomized rank-k approximation of an interpolative decomposition of the input matrix."""
    p = k + oversampling
    if(p <= k or p > A.shape[1]):
        raise ValueError('Invalid p')
    cols = np.random.choice(A.shape[1], replace=False, size=p)
    AS = A[:,cols]
    Q, R, P = scipy.linalg.qr(AS, pivoting=True, mode=full_m)
    Q = Q[:,:k]
    return Q @ Q.T @ A