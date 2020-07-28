import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import scipy.linalg


def svd_rank_k(A, k):
    """Calculates a deterministic rank-k approximation of a singular value decomposition of the input matrix."""
    U, sigma, Vh = np.linalg.svd(A, full_matrices=False)
    Sigma = np.diag(sigma[:k])
    return U[:,:k] @ Sigma @ Vh[:k]


def random_svd_rank_k(A, k, oversampling=10, power=1, only_basis=False):
    """Calculates a randomized rank-k approximation of a singular value decomposition of the input matrix."""
    if oversampling <= 0:
        raise ValueError('Oversampling parameter must be greater than zero.')
    p = k + oversampling
    if p > A.shape[1]:
        raise ValueError('Oversampling parameter is too large.')
       
    Omega = random.randn(A.shape[1], k)
    Y = A @ Omega
    if power != 0:
        Y = np.linalg.matrix_power(A @ A.T, power) @ Y
    Q, R = np.linalg.qr(Y)
    
    B = Q.T @ A # warning: we assume Q is real-valued
    U_tilde, sigma, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    Sigma = np.diag(sigma[:k])
    
    if only_basis:
        return U[:,:k]
    else:
        return U[:,:k] @ Sigma @ Vh[:k]


def id_rank_k(A, k):
    """Calculates a deterministic rank-k approximation of an interpolative decomposition of the input matrix."""
    Q, R, P = scipy.linalg.qr(A, pivoting=True, mode='economic')
    Q = Q[:,:k]
    return Q @ Q.T @ A


def random_id_rank_k(A, k, oversampling=10):
    """Calculates a randomized rank-k approximation of an interpolative decomposition of the input matrix."""
    if oversampling <= 0:
        raise ValueError('Oversampling parameter must be greater than zero.')
    p = k + oversampling
    if p > A.shape[1]:
        raise ValueError('Oversampling parameter is too large.')
    cols = np.random.choice(A.shape[1], replace=False, size=p)
    AS = A[:,cols]
    Q, R, P = scipy.linalg.qr(AS, pivoting=True, mode='economic')
    Q = Q[:,:k]
    return Q @ Q.T @ A