import numpy as np


def blockCG(A, X, B, max_iter=1, tol=1e-3):
    """Solve multiple linear systems AX=B using the block conjugate gradient (BCG) method described by O'Leary (1980).

    Parameters
    ----------
    A : ndarray of shape ()
        Linear operator (LHS)

    X0 : ndarray of shape ()
        Initial guess

    B : ndarray of shape ()
        Right hand side

    max_iter : int, default=1
        Maximum number of iterations

    tol : float, default=1e-3
        Convergence threshold

    Returns
    -------
    X : ndarray of shape ()
        Solution matrix

    References
    ----------
    D. P. O'Leary, “The block conjugate gradient algorithm and related methods,”
        Linear Algebra and its Applications, vol. 29, pp. 293-322, 2 1980
    """
    # Calculate residual
    R = B - A.dot(X)
    if R < tol:
        print(f"Guessed solution satisifes threshold.")
        return X

    # Iteratively calculate solution matrix X
    P = R
    for ii in range(max_iter):
        print(f"Iteration {ii+1}...")

        # Update X
        PtAP = P.T.dot(A.dot(P))
        L = np.linalg.pinv(PtAP).dot(R.T.dot(R))
        X += P.dot(L)

        # Recalculate residual
        R -= A.dot(P.dot(L))

        # Check convergence
        if R < tol:
            print(f"Converged at iteration {ii+1}.")
            return X
        else:
            Phi = np.linalg.pinv(R.dot(R)).dot(R.T.dot(R))
            P = R + P.dot(Phi)

    return X
