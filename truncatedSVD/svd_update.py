import numpy as np
from scipy.linalg import block_diag


def zha_simon_update(A, Uk, Sk, VHk, E):
    """Calculate truncated SVD update using Zha-Simon projection algorithm.

    Parameters
    ----------
    A : array, shape (m, n)
        Update matrix

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
    # Construct Z and ZH*A matrices
    s = E.shape[0]
    k = Uk.shape[1]
    Z = block_diag(Uk, np.eye(s))
    ZHA = np.vstack((np.diag(Sk).dot(VHk), E))

    # # kSVD of ZH*A
    # # TODO: implement unrestarted Lanczos method on ZH*A*AH*Z
    # print("Performing kSVD on ZH*A.")
    # (F_matrix, Theta_array, G_matrix) = np.linalg.svd(np.block(
    #   [[ np.dot(np.diag(self.Sigmak_array), self.VHk_matrix) ],
    #    [ self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:] ]]
    # ), full_matrices=False)

    Fk, Tk, _ = np.linalg.svd(ZHA, full_matrices=False)

    # Truncate if necessary
    if k < len(Tk):
        Fk = Fk[:, :k]
        Tk = Tk[:k]

    # Calculate updated values for Uk, Sk, Vk
    Uk_new = Z.dot(Fk)
    Vk_new = A.T.dot(Uk_new.dot(np.diag(1 / Tk)))

    return Uk_new, Tk, Vk_new.T


def bcg_update():
    """Calculate truncated SVD update using enhanced projection matrix."""
    # TODO: implement block CG update
    return None


def brute_force_update(A, k, full_matrices=False):
    """Calculate best rank-k approximation using brute force."""
    # TODO: implement brute force update
    _, Sk, VHk = np.linalg.svd(A.T.dot(A), full_matrices=full_matrices)
    return VHk[:k, :], Sk[:k]


def naive_update():
    """Calculate naive update."""
    # TODO: implement naive update
    return None
