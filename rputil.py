import numpy as np
from numpy import random
import math
from matplotlib import pyplot as plt
import scipy.linalg

def random_svd_rank_k(A, k, power=1):
    omega = random.randn(A.shape[1],k)
    Y = np.linalg.matrix_power(A @ A.T, power) @ (A @ omega)
    Q, R = np.linalg.qr(Y)
    B = Q.T @ A
    U_tilde, Sigma, Vh = np.linalg.svd(B)
    U = Q @ U_tilde
    Sigma = np.diag(Sigma)
    return U @ Sigma @ Vh[:k]

def svd_rank_k(matrix, k):
    u, sigma, vh = np.linalg.svd(matrix) #compute full svd
    u = u[:,:k] #keep all rows, take first k columns
    sigma = np.diag(sigma[:k]) #take first k singular values and make diagonal matrix
    vh = vh[:k] #take first k rows, take all columns
    return u @ sigma @ vh #return rank k approximation

def id_rank_k(matrix, k):
    q, r, p = scipy.linalg.qr(matrix, pivoting=True)
    q = q[:,:k]
    return q @ q.T @ matrix

def random_id_rank_k(matrix, k, oversampling=10):
    p = k + oversampling
    if(p<=k or p>matrix.shape[1]):
        print('Invalid p')
        return False
    cols = np.random.choice(matrix.shape[1], replace = False, size = p)
    AS = matrix[:,cols]
    q,r = np.linalg.qr(AS)
    q = q[:,:k]
    return q @ q.T @ matrix

def proj(v,u): #projection of v onto u
    return (np.dot(v,u) / np.dot(u,u)) * u

def id_rank_k_slow(matrix, k):
    
    m = np.copy(matrix) #starts out as full matrix
    indices = []
        
    if False:
        indices = predetermined[:k]
    else:
        #Greedy determinant maximization process
        for j in range(k):
            col_norms = np.zeros(matrix.shape[1])
            for ind in np.delete(np.arange(matrix.shape[1]), indices): #loop through non-chosen columns
                col_norms[ind] = np.linalg.norm(m[:,ind]) #gets norms of non-chosen columns
            max_norm_index = np.argmax(col_norms) #index of column with greatest l2 norm

            indices.append(max_norm_index) #add the index to a list of indices

            #orthogonalize columns with all other indices
            for i in np.delete(np.arange(matrix.shape[1]), indices):   #iterate over all other indices and do g-s step
                for ind in indices:
                    m[:,i] = m[:,i] - proj(m[:,i],m[:,ind])

    Ak = matrix[:,indices] # first k columns of A
    Other = matrix[:,np.delete(np.arange(matrix.shape[1]),indices)] # other columns of A
    AP = np.concatenate((Ak,Other),axis=1) # combine it so we have the data, A, reordered, with the 'best' k coming first

    q,r = np.linalg.qr(AP) #now compute QR
    q = q[:,:k] # and take the first k, corresponding to the 'best' k from A

    return q @ q.T @ matrix
    