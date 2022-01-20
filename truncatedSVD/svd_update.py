"""Various algorithms for updating the truncated singular value decomposition (SVD) of evolving matrices.
"""

import numpy as np
import time
from scipy.linalg import block_diag
from blockCG import blockCG
from sklearn.utils.extmath import randomized_svd as rsvd


def zha_simon_update(A, Uk, Sk, VHk, E):
    """Calculate truncated SVD update using Zha-Simon projection algorithm.

    Parameters
    ----------
    A : array, shape (m, n)
        Updated matrix

    Uk : array, shape (m, k)
        Left singular vectors from previous update

    Sk : array, shape (k,)
        Singular values from previous update

    VHk : array, shape (k, n)
        Right singular vectors from previous update

    E : array, shape (s, n)
        Appended submatrix

    Returns
    -------
    Uk_new : array, shape (m, k)
        Updated left singular vectors

    Sk_new : array, shape (k,)
        Updated singular values

    VHk_new : array, shape (k, n)
        Update right singular vectors

    References
    ----------
    H. Zha and H. D. Simon, “Timely communication on updating problems in latent semantic indexing,
        ”Society for Industrial and Applied Mathematics, vol. 21, no. 2, pp. 782-791, 1999.
    """
    print("Updating truncated SVD using Zha-Simon projection method.")

    # Construct Z and ZH*A matrices
    s = E.shape[0]
    k = Uk.shape[1]
    Z = block_diag(Uk, np.eye(s))
    ZHA = np.vstack((np.diag(Sk).dot(VHk), E))

    # Calculate SVD of ZH*A
    Fk, Tk, _ = np.linalg.svd(ZHA, full_matrices=False)

    # Truncate if necessary
    if k < len(Tk):
        Fk = Fk[:, :k]
        Tk = Tk[:k]

    # Calculate updated values for Uk, Sk, Vk
    Uk_new = Z.dot(Fk)
    Vk_new = A.T.dot(Uk_new.dot(np.diag(1 / Tk)))

    return Uk_new, Tk, Vk_new.T


def bcg_update(B, Uk, sigmak, VHk, E, lam=None, r=10):
    """Calculate truncated SVD update using enhanced projection matrix.

    Parameters
    ----------
    B : ndarray of shape (m, n)
        Current matrix

    Uk : ndarray of shape (m, k)
        Left singular vectors

    sigmak : ndarray of shape (k,)
        Singular values

    VHk : ndarray of shape (n, k)
        Right singular vectors

    E : ndarray of shape (s, n)
        Matrix to be appended

    lam : float, default=None
        If 'None', lam is set to 1.01 * (sigmahat_1)^2

    r : int, default=10
        Parameter determining number of columns in matrix R

    Returns
    -------
    Uk_new : array, shape (m, k)
        Updated left singular vectors

    Sk_new : array, shape (k,)
        Updated singular values

    VHk_new : array, shape (k, n)
        Update right singular vectors

    References
    ----------
    H. Zha and H. D. Simon, “Timely communication on updating problems in latent semantic indexing,
        ”Society for Industrial and Applied Mathematics, vol. 21, no. 2, pp. 782-791, 1999.
    """
    print("Updating truncated SVD using enhanced projection method.")
    k = len(sigmak)

    # Set lambda
    if lam is None:
        lam = 1.01 * sigmak[0] ** 2

    # Calculate B(lambda) B E^H
    print("Calculating B(lambda) B E^H using BCG.")
    m = B.shape[0]
    lhs = -(B.dot(B.T) - lam * np.eye(m))
    rhs = (np.eye(m) - Uk.dot(Uk.T)).dot(B.dot(E.T))
    BlBEH = blockCG(lhs, rhs, max_iter=1)

    # Calculate B(lambda) B E^H R
    BlBEHR = BlBEH.dot(np.random.normal(size=(E.shape[0], 2 * r)))

    # Calculate X_lambda_r using randomized SVD
    print("Calculating X_lambda_r.")
    # Xlr, _, _ = np.linalg.svd(BlBEHR, full_matrices=False)
    # Xlr = Xlr[:, :r]
    Xlr, _, _ = rsvd(BlBEH, n_components=r, n_oversamples=r, n_iter=0)

    # Construct Z matrix
    print("Constructing Z matrix.")
    Z = block_diag(np.hstack((Uk, Xlr)), np.eye(E.shape[0]))

    # Construct ZH*A matrix
    ZHA = np.vstack((np.diag(sigmak).dot(VHk), Xlr.T.dot(B), E))

    # Calculate SVD of ZH*A
    Fk, Tk, _ = np.linalg.svd(ZHA, full_matrices=False)

    # Truncate if necessary
    if k < len(Tk):
        Fk = Fk[:, :k]
        Tk = Tk[:k]

    # Calculate updated values for Uk, Sk, Vk
    Uk_new = Z.dot(Fk)
    A = np.vstack((B, E))
    Vk_new = A.T.dot(Uk_new.dot(np.diag(1 / Tk)))

    return Uk_new, Tk, Vk_new.T
    

def brute_force_update(A, k, full_matrices=False):
    """Calculate best rank-k approximation using brute force.
    
    Parameters
    ----------
    A : ndarray of shape (m, n)
    
    k : int
        Desired rank of approximation
    
    full_matrices : bool, default=False
        Option to return full matrices
    
    Returns 
    -------
    VHk : ndarray of shape (n, k)
        Truncated right singular vectors
    
    Sk : ndarray of shape (k,)
        Truncated singular values
    
    References
    ----------
    M. Ghashami, E. Liberty, J. M. Phillips, and D. P. Woodruff, 
        “Frequent Directions: Simple and Deterministic Matrix Sketching,
        ”SIAM Journal on Computing, vol. 45, no. 5, pp. 1762-1792, 1 2016
    """
    # TODO: implement brute force update
    _, Sk, VHk = np.linalg.svd(A.T.dot(A), full_matrices=full_matrices)
    return VHk[:k, :], Sk[:k]


def naive_update(l, d):
    """Calculate naive update. Given the updated matrix, returns a matrix of zeros.
    
    Parameters
    ----------
    A : ndarray of shape ()
        Updated matrix
        
    l : int
        Number of rows appended
    
    Returns
    -------
    zeros : ndarray of shape (l, d)
        Zeros of shape (l, d)
    
    References
    ----------
    M. Ghashami, E. Liberty, J. M. Phillips, and D. P. Woodruff, 
        “Frequent Directions: Simple and Deterministic Matrix Sketching,
        ”SIAM Journal on Computing, vol. 45, no. 5, pp. 1762-1792, 1 2016
    """    
    return np.zeros((l, d))


def fd_update(fd, E):
    """Calculate truncated SVD update using Frequent Directions algorithm.
    
    Parameters
    ----------
    fd : FrequentDirections object
        FrequentDirections object used for performing FD updates
    
    E : array, shape (s, n)
        Appended submatrix

    Returns
    -------
    fd : FrequentDirections object
        FrequentDirections object used for performing FD updates
    
    References
    ----------
    M. Ghashami, E. Liberty, J. M. Phillips, and D. P. Woodruff, 
        “Frequent Directions: Simple and Deterministic Matrix Sketching,
        ”SIAM Journal on Computing, vol. 45, no. 5, pp. 1762-1792, 1 2016
    """    
    for row in E:
      fd.append(row)
    
    return fd
