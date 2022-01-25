import numpy as np
from sklearn.metrics import mean_squared_error


def query_precision_recall(A, Q, relevant_docs):
    """Return precision-recall curve for LSI application
    
    Wrapper for sklearn implementation of computing precision-recall pairs.
    
    Parameters
    ----------
    A : ndarray of shape (n_terms, n_docs)
         Term-document matrix
        
    Q : ndarray of shape (n_queries, n_terms)
        Query matrix
    
    relevant_docs : ndarray of shape (n_queries, n_)
    
    Returns
    -------
    precision : ndarray of shape (,)
        11-point interpolated precision
    
    recall : ndarray of shape (,)
        Recall
    
    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve
    """
    # Check dimensions of query and term-document matrices
    assert Q.shape[1] == A.shape[0]
    
    # Calculate scores for each query
    scores = Q.dot(A)
    
    # Calculate 11-point average precision
    precision = 0
    recall = 0
    
    return precision, recall


def mse(y_true, y_pred):
    """Return mean squared error
    
    Parameters
    ----------
    y_true : ndarray of shape (n,)
        True labels
        
    y_pred : ndarray of shape (n,)
        Predicted labels
        
    Returns
    -------
    loss : float
        Mean squared error
    """
    return mean_squared_error(y_true, y_pred)


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

    Notes
    -----
    As described by Ghashami et al., the value of 'Ahat' can be calculated
    using either of the definitions listed.
    '"""
    return (np.linalg.norm(A - Ahat, ord="fro") ** 2) / (
        np.linalg.norm(A - Ak, ord="fro") ** 2
    )


def cov_err(A, B):
    """Calculate covariance error

    Calculate covariance error as defined by Ghashami et al. (2016).

    Parameters
    ----------
    A : array, shape (n, d)
        Original data matrix

    B : array, shape (l, d)
        Sketch matrix

    Returns
    -------
    cov_error : float
        Covariance error

    References
    ----------
    M. Ghashami, E. Liberty, J. M. Phillips, and D. P. Woodruff,
      “Frequent Directions: Simple and Deterministic Matrix Sketching,"
      SIAM Journal on Computing, vol. 45, no. 5, pp. 1762-1792, 1 2016.
    """
    return np.linalg.norm(A.T.dot(A) - B.T.dot(B)) / np.linalg.norm(A, ord="fro") ** 2


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
    """
    return np.linalg.norm(A.dot(V) - U * s, axis=0) / s
