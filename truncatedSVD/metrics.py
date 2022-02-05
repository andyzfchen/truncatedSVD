import numpy as np


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
