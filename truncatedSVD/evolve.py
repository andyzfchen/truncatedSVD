import numpy as np
from scipy.linalg import block_diag


def zha_simon_update(A, Uk, Sk, Vk, E):
    """Calculate truncated SVD update using Zha-Simon projection algorithm.
    
    Parameters
    ----------
    A : array, shape ()
        Update matrix
    
    Uk : array, shape ()
        Left singular vectors from previous update
    
    Sk : array, shape ()
        Singular values from previous update
    
    Vk : array, shape ()
        Right singular vectors from previous update
        
    E : array, shape ()
        Appended submatrix
        
    Returns
    -------
    Uk_new : array, shape ()
        Updated left singular vectors
    
    Sk : array, shape ()
        Updated singular values

    Vk : array, shape ()
        Update right singular vectors
        
    References
    ----------
    H. Zha and H. D. Simon, “Timely communication on updating problems in latent semantic indexing,
    ”Society for Industrial and Applied Mathematics, vol. 21, no. 2, pp. 782–791, 1999.
    """
    s = E.shape[0]
    Z = block_diag(Uk, np.eye(s))
    ZHA = np.vstack((Sk.dot(Vk.T), E))
    Fk, Tk, _ = np.linalg.svd(ZHA, full_matrices=False)
    
    # Calculate updated values for Uk, Sk, Vk
    Uk_new = Z.dot(Fk)
    Vk_new = A.T.dot(Uk_new.dot(np.diag(1 / Tk)))
    return Uk_new, Tk, Vk_new.T


def bcg():
    """Calculate truncated SVD update using enhanced projection matrix.
    """
    return None
