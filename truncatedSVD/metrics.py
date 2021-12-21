import numpy as np


def proj_err(A, Ahat, Ak):
    """Calculate projection error
    
    Calculate projection error as defined by Ghashami et al. (2016).
    
    Parameters
    ----------
    A : array, shape (n, d)
        Input matrix

    Ahat : array, shape (n, d)
        1. Projection of A onto the best rank-k approximation to B
        2. Best rank-k approximation of projection of A onto B

    Ak : array, shape (n, d)
        Best rank-k approximation of A
    
    Returns
    -------
    proj_error : float
        Projection error
    
    References
    ----------
    M. Ghashami, E. Liberty, J. M. Phillips, and D. P. Woodruff, 
    “Frequent Directions: Simple and Deterministic Matrix Sketching,
    SIAM Journal on Computing, vol. 45, no. 5, pp. 1762-1792, 1 2016.
    [Online]. Available:http://epubs.siam.org/doi/10.1137/15M1009718
    
    Notes
    -----
    As described by Ghashami et al., the value of 'Ahat' can be calculated 
    using either of the definitions listed.
    '"""
    return (np.linalg.norm(A - Ahat, ord='fro') ** 2) / (np.linalg.norm(A - Ak, ord='fro') ** 2)


def cov_err(A, B):
    """Calculate covariance error
    
    Calculate covariance error as defined by Ghashami et al. (2016).
    
    Parameters
    ----------
    A : array, shape (n, d)
    
    B : array, shape (n, d)
    
    Returns
    -------
    cov_error : float
        Covariance error
    
    References
    ----------
    M. Ghashami, E. Liberty, J. M. Phillips, and D. P. Woodruff, 
    “Frequent Directions: Simple and Deterministic Matrix Sketching,"
    SIAM Journal on Computing, vol. 45, no. 5, pp. 1762-1792, 1 2016.
    [Online]. Available:http://epubs.siam.org/doi/10.1137/15M1009718 
    """
    return np.linalg.norm(A.T.dot(A) - B.T.dot(B)) / np.linalg.norm(A, ord='fro') ** 2


def rel_err(sv, sv_hat):
    """
    Calculate relative error of singular values as defined by Kalantzis et al. (2021).
    
    Parameters
    ----------
    sv : array, shape (k,)
        True singular values
        
    sv_hat : array, shape (k,)
        Approximated singular values

    Returns
    -------
    rel_error : array, shape (n,)
        Relative error of singular values
        
    References
    ----------
    V. Kalantzis, G. Kollias, S. Ubaru, A. N. Nikolakopoulos, L. Horesh, and K. L. Clarkson, 
    “Projection techniquesto update the truncated SVD of evolving matrices with applications,” 
    inProceedings of the 38th InternationalConference on Machine Learning, 
    M. Meila and T. Zhang, Eds.PMLR, 7 2021, pp. 5236-5246.
    [Online].Available: https://proceedings.mlr.press/v139/kalantzis21a.html
    """
    return np.abs(sv_hat - sv) / sv


def res_norm(A, U, V, s):
    """
    Calculate scaled residual norm as defined by Kalantzis et al. (2021). 

    Parameters
    ----------
    A : array, shape (m, n)
        Original data matrix
        
    U : array, shape (m, k)
        Left singular vectors
    
    V : array, shape (n, k)
        Right singular vectors

    s : array, shape (k,)
        Singular values
    
    Returns
    -------
    res_norm : array, shape (k,)
        Scaled residual norm

    References
    ----------
    V. Kalantzis, G. Kollias, S. Ubaru, A. N. Nikolakopoulos, L. Horesh, and K. L. Clarkson, 
    “Projection techniquesto update the truncated SVD of evolving matrices with applications,” 
    inProceedings of the 38th InternationalConference on Machine Learning, 
    M. Meila and T. Zhang, Eds.PMLR, 7 2021, pp. 5236-5246.
    [Online].Available: https://proceedings.mlr.press/v139/kalantzis21a.html
    """
    return np.linalg.norm(A.dot(V) - U * s, axis=0) / s
