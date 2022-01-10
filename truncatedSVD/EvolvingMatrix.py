"""Evolving matrix class for updating truncated singular value decomposition (SVD) of evolving matrices.
"""

import numpy as np
import scipy.sparse.linalg
import sklearn.decomposition
import os
import time
from metrics import proj_err, cov_err, rel_err, res_norm
from svd_update import zha_simon_update, bcg_update, brute_force_update, naive_update


class EvolvingMatrix(object):
    """Evolving matrix with periodically appended rows.

    This class simulates a matrix subject to the periodic addition of new rows.
    Applications with such matrices include latent semantic indexing (LSI) and recommender systems.

    Given an initial matrix, rows from a matrix to be appended are added in sequential updates.
    With each batch update, the truncated singular value decomposition (SVD)
    of the new matrix can be calculated using a variety of methods.
    The accuracy of these methods can be evaluated by four metrics.

    Parameters
    ----------
    initial_matrix : ndarray of shape (m, n)
        Initial matrix

    append_matrix : nd array of shape (s, n)
        Entire matrix to be appended row-wise to initial matrix

    n_batches : int, default=1
        Number of batches

    k_dim : int, default=50
        Rank of truncated SVD to be calculated

    Attributes
    ----------
    U_true, VH_true, S_true : ndarrays of shape ()
        True SVD of current update

    Uk, VHk, Sk : ndarrays of shape ()
        Truncated SVD calculated using one of the methods

    runtime : float
        Total time elapsed in calculating updates so far    
    
    References
    ----------
    H. Zha and H. D. Simon, “Timely communication on updating problems in latent semantic indexing,
        ”Society for Industrial and Applied Mathematics, vol. 21, no. 2, pp. 782-791, 1999.

    V. Kalantzis, G. Kollias, S. Ubaru, A. N. Nikolakopoulos, L. Horesh, and K. L. Clarkson,
        “Projection techniquesto update the truncated SVD of evolving matrices with applications,”
        inProceedings of the 38th InternationalConference on Machine Learning,
        M. Meila and T. Zhang, Eds.PMLR, 7 2021, pp. 5236-5246.
    """

    def __init__(self, initial_matrix, append_matrix=None, n_batches=1, k_dim=None):
        # Initial matrix
        self.initial_matrix = initial_matrix.toarray()  # ensure data is in dense format
        (self.m_dim, self.n_dim) = np.shape(self.initial_matrix)
        print(
            f"Initial matrix of evolving matrix set to shape of ( {self.m_dim}, {self.n_dim} )."
        )

        # Update matrix after each batch
        self.A = self.initial_matrix

        # True SVD of current update - initialized to SVD of the initial matrix
        self.U_true, self.sigma_true, self.VH_true = np.linalg.svd(
            self.initial_matrix, full_matrices=False
        )

        # Truncated SVD of current update - initialized to rank-k SVD of the initial matrix
        if k_dim is None:
            self.k_dim = min(self.m_dim, self.n_dim)
        else:
            self.k_dim = k_dim

        self.Uk = self.U_true[:, : self.k_dim]
        self.sigmak = self.sigma_true[: self.k_dim]
        self.VHk = self.VH_true[: self.k_dim, :]

        print(
            f"Initial Uk  matrix of evolving matrix set to shape of ( {np.shape(self.Uk)} )."
        )
        print(
            f"Initial Sigmak matrix of evolving matrix set to shape of ( {np.shape(self.sigmak)} )."
        )
        print(
            f"Initial VHk matrix of evolving matrix set to shape of ( {np.shape(self.VHk)} )."
        )

        # Initialize matrix to be appended
        self.n_batches = n_batches
        if append_matrix is None:
            self.U_all, self.sigma_all, self.VH_all = (
                np.array([]),
                np.array([]),
                np.array([]),
            )
            self.append_matrix = np.array([])
            self.s_dim = 0
            self.step = 0
        else:
            self.U_all, self.sigma_all, self.VH_all = self.set_append_matrix(
                append_matrix
            )

        # Initialize submatrix to be appended at each update
        self.update_matrix = np.array([])

        # Initialize parameters to keep track of updates
        self.phi = 0
        self.n_appended = 0
        self.n_appended_total = 0

        # Initialize total runtime
        self.runtime = 0.0

    def set_append_matrix(self, append_matrix):
        """Initialize entire matrix to append E"""
        self.append_matrix = append_matrix.toarray()  # ensure data is in dense format
        (self.s_dim, n_dim) = np.shape(self.append_matrix)
        self.step = int(np.ceil(self.s_dim / self.n_batches))
        assert n_dim == self.n_dim
        print(f"Appendix matrix set to shape of ( {self.s_dim} , {self.n_dim} ).")

        self.U_all, self.Sigma_all, self.VH_all = np.linalg.svd(
            np.append(self.initial_matrix, self.append_matrix, axis=0)
        )

    def evolve(self):
        """Evolve matrix by one update according to update parameters specified."""
        # Check if number of appended rows exceeds number of remaining rows in appendix matrix
        self.n_appended = (
            self.step
            if self.step <= (self.s_dim - self.n_appended_total)
            else self.s_dim - self.n_appended_total
        )

        # Append to current data matrix
        print(f"Appending {self.n_appended} rows from appendix matrix.")
        self.update_matrix = self.append_matrix[
            self.n_appended_total : self.n_appended_total + self.n_appended, :
        ]
        self.A = np.append(self.A, self.update_matrix, axis=0)

        # Update counters for update
        self.phi += 1
        self.n_appended_total += self.n_appended
        print(
            f"Appended {self.n_appended_total}/{self.s_dim} rows from appendix matrix so far."
        )

    def reset(self):
        """Undo all evolutions and set current matrix to the initial matrix. Matrix to be appended is unchanged."""
        self.A_matrix = self.initial_matrix
        self.update_matrix = np.array([])
        self.n_appended = 0
        self.n_appended_total = 0

    def update_svd_zha_simon(self):
        """Return truncated SVD of updated matrix using the Zha-Simon projection method."""
        start = time.perf_counter()
        self.Uk, self.sigmak, self.VHk = zha_simon_update(
            self.A, self.Uk, self.sigmak, self.VHk, self.update_matrix
        )
        self.runtime += time.perf_counter() - start
        return self.Uk, self.sigmak, self.VHk

    def update_svd_bcg(self):
        """Return truncated SVD of updated matrix using the BCG method."""
        # Get previous data matrix from updated matrix
        B = self.A[: -self.n_appended, :]

        # Update truncated SVD
        start = time.perf_counter()
        self.Uk, self.sigmak, self.VHk = bcg_update(
            B, self.Uk, self.sigmak, self.VHk, self.update_matrix
        )
        self.runtime += time.perf_counter() - start
        return self.Uk, self.sigmak, self.VHk

    def update_svd_brute_force(self):
        """Return optimal rank-k approximation of updated matrix using brute force method."""
        self.Uk, self.sigmak, self.VHk = brute_force_update(
            self.A, self.Uk, self.sigmak, self.VHk, self.update_matrix
        )
        return self.Uk_matrix, self.Sigmak_array, self.VHk_matrix

    def update_svd_naive(self):
        """Return truncated SVD of updated matrix using the naïve update method."""
        self.Uk, self.sigmak, self.VHk = naive_update(
            self.A, self.Uk, self.sigmak, self.VHk, self.update_matrix
        )
        return self.Uk, self.sigmak, self.VHk

    def evolve_matrix_deflated_bcg(self, step_dim=None, r_dim=None):
        """Construct Z using enhanced projection method.

        Z = [[U_k, X_lambda,r 0] [0 I_s]]
        """
        print("Using deflated BCG method to evolve matrix.")

        # default number of appended rows
        if step_dim is None:
            step_dim = self.step_dim

        # default r dimension
        if r_dim is None:
            r_dim = 10

        # checks if number of appended rows exceeds number of remaining rows in appendix matrix
        if step_dim > (self.s_dim - self.n_rows_appended):
            step_dim = self.s_dim - self.n_rows_appended

        # checks if Xlambdar has full r rank for Z matrix construction
        if r_dim > step_dim:
            r_dim = step_dim

        # Xlambdar
        print("Calculating X matrix.")
        E_matrix = self.append_matrix[
            self.n_rows_appended : self.n_rows_appended + step_dim, :
        ]
        svd = sklearn.decomposition.TruncatedSVD(n_components=5, n_iter=20)
        svd.fit(
            np.append(
                self.A_matrix,
                self.append_matrix[
                    self.n_rows_appended : self.n_rows_appended + step_dim, :
                ],
                axis=0,
            )
        )
        # lambda_value = 1.01 * self.Sigmak_array[0]**2   # lambda_value should be >= first singular value
        lambda_value = (
            1.01 * svd.singular_values_[0] ** 2
        )  # lambda_value should be >= first singular value
        LHS_matrix = -(
            np.dot(self.A_matrix, self.A_matrix.T)
            - lambda_value * np.eye(self.m_dim + self.n_rows_appended)
        )
        RHS_matrix = np.dot(
            (
                np.eye(self.m_dim + self.n_rows_appended)
                - np.dot(self.Uk_matrix, self.Uk_matrix.T)
            ),
            np.dot(self.A_matrix, E_matrix.T),
        )

        """
    X_matrix = np.zeros((self.m_dim+self.n_rows_appended, step_dim))
    # TODO: currently using CG column by column; need to implement block CG instead
    for ii in range(step_dim):
      if (ii+1)%50 == 0:
        print("Step "+str(ii+1)+" of "+str(step_dim)+".")
      X_matrix[:,ii] = scipy.sparse.linalg.cg(LHS_matrix, RHS_matrix[:,ii])[0]
    print("Inverting matrix for X matrix.")
    """
        X_matrix = np.dot(np.linalg.inv(LHS_matrix), RHS_matrix)

        # rSVD of X
        print("Performing randomized rSVD on X for X_lambda,r matrix.")
        X_matrix = np.dot(X_matrix, np.random.normal(size=(step_dim, 2 * r_dim)))
        (Xlambdar_matrix, Slambdar_array, Ylambdar_matrix) = np.linalg.svd(
            X_matrix, full_matrices=False
        )

        # Z matrix
        Z_matrix = np.block(
            [
                [
                    self.Uk_matrix,
                    Xlambdar_matrix[:, :r_dim],
                    np.zeros((self.m_dim + self.n_rows_appended, step_dim)),
                ],
                [np.zeros((step_dim, self.k_dim + r_dim)), np.eye(step_dim)],
            ]
        )

        # kSVD of ZH*A    # TODO: implement unrestarted Lanczos method on ZH*A*AH*Z
        print("Performing kSVD on ZH*A.")
        ZHA = np.block(
            [
                [np.dot(np.diag(self.Sigmak_array), self.VHk_matrix)],
                [np.dot(Xlambdar_matrix[:, :r_dim].T, self.A_matrix)],
                [
                    self.append_matrix[
                        self.n_rows_appended : self.n_rows_appended + step_dim, :
                    ]
                ],
            ]
        )
        """
    ZHA = np.block([ np.dot(self.VHk_matrix.T, np.diag(self.Sigmak_array)), np.dot(self.A_matrix.T, Xlambdar_matrix[:,:r_dim]), self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:].T ]).T
    """
        (F_matrix, Theta_array, G_matrix) = np.linalg.svd(ZHA, full_matrices=False)

        print(
            "Appending ",
            step_dim,
            " rows from appendix matrix and evolving truncated matrix.",
        )

        self.A_matrix = np.append(
            self.A_matrix,
            self.append_matrix[
                self.n_rows_appended : self.n_rows_appended + step_dim, :
            ],
            axis=0,
        )

        self.n_rows_appended += step_dim

        print(
            "Appended ",
            self.n_rows_appended,
            " of ",
            self.s_dim,
            " rows from appendix matrix so far.",
        )

        # recalculation of Uk, Sigmak, and VHk
        print("Recalculating U_k, Sigma_k, and VH_k.")
        self.Uk_matrix = np.dot(Z_matrix, F_matrix[:, : self.k_dim])
        self.Sigmak_array = Theta_array[: self.k_dim]
        self.VHk_matrix = np.dot(
            np.dot(self.A_matrix.T, self.Uk_matrix), np.diag(1 / self.Sigmak_array)
        ).T

        print()

        return self.Uk_matrix, self.Sigmak_array, self.VHk_matrix

    def calculate_true_svd(self, evolution_method, dataset):
        """Calculate true SVD of the current A matrix."""
        print("Calculating current A matrix of size ", np.shape(self.A))

        # Folder to save U,S,V arrays
        dirname = (
            f"../cache/{evolution_method}/{dataset}_batch_split_{str(self.n_batches)}"
        )

        # Load from cache if pre-calculated
        if os.path.exists(f"{dirname}/U_true_phi_{str(self.phi)}.npy"):
            print("Loading from cache")
            self.U_true = np.load(f"{dirname}/U_true_phi_{str(self.phi)}.npy")
            self.sigma_true = np.load(f"{dirname}/sigma_true_phi_{str(self.phi)}.npy")
            self.VH_true = np.load(f"{dirname}/VH_true_phi_{str(self.phi)}.npy")
        else:
            self.U_true, self.sigma_true, self.VH_true = np.linalg.svd(self.A)
            np.save(f"{dirname}/U_true_phi_{str(self.phi)}.npy", self.U_true)
            np.save(f"{dirname}/sigma_true_phi_{str(self.phi)}.npy", self.sigma_true)
            np.save(f"{dirname}/VH_true_phi_{str(self.phi)}.npy", self.VH_true)

    def get_mean_squared_error(self):
        return None

    def get_relative_error(self, sv_idx=None):
        """Return relative error of n-th singular value"""
        if sv_idx is None:
            sv_idx = np.arange(self.k_dim)

        return rel_err(self.sigma_true[sv_idx], self.sigmak[sv_idx])

    def get_residual_norm(self, sv_idx=None):
        """Return residual norm of n-th singular vector"""
        if sv_idx is None:
            sv_idx = np.arange(self.k_dim)

        return res_norm(
            self.A, self.Uk[:, sv_idx], self.VHk[sv_idx, :].T, self.sigmak[sv_idx]
        )

    def get_covariance_error(self):
        """Return covariance error"""
        return cov_err(
            self.A, self.U_true.dot(np.diag(self.sigma_true).dot(self.VH_true))
        )

    def get_projection_error(self):
        """Return projection error"""
        return proj_err()

    def save_metrics(self, fdir, print_metrics=True, sv_idx=None, r_str=""):
        """Calculate and save metrics and optionally print to console."""
        # Calculate metrics
        rel_err = self.get_relative_error(sv_idx=sv_idx)
        res_norm = self.get_residual_norm(sv_idx=sv_idx)
        # cov_err = self.get_covariance_error()
        # proj_err = self.get_projection_error()

        # Save metrics
        np.save(f"{fdir}/relative_errors_phi_{self.phi}{r_str}.npy", rel_err)
        np.save(f"{fdir}/residual_norms_phi_{self.phi}{r_str}.npy", res_norm)
        # np.save(f"{fdir}/covariance_error_phi_{self.phi}{r_str}.npy", cov_err)
        # np.save(f"{fdir}/projection_error_phi_{self.phi}{r_str}.npy", proj_err)
        np.save(f"{fdir}/runtime_phi_{self.phi}{r_str}.npy", self.runtime)

        # Optionally print metrics to console
        if print_metrics:
            print(
                f"Singular value relative errors at phi = {str(self.phi)}:\n{rel_err}"
            )
            print(
                f"Last singular vector residual norm at phi = {str(self.phi)}:\n{res_norm}"
            )
            # print(f"Covariance error at phi = {str(self.phi)}:\n{cov_err}")
            # print(f"Projection error at phi = {str(self.phi)}:\n{proj_err}")
