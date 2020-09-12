# Random Projections is a collection of work done by the authors during the Summer@ICERM 2020 REU program.
# Copyright (C) 2020  Rishi Advani, Madison Crim, Sean O'Hagan

# This file is part of Random Projections.

# Random Projections is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Random Projections is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Random Projections.  If not, see <https://www.gnu.org/licenses/>.



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