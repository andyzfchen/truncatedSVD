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
    
    append_matrix : ndarray of shape ()
        Entire matrix to be appended over the course of updates
    
    update_matrix : ndarray of shape ()
        Matrix appended in last update
    
    n_appended : int
        Number of rows appended in last update
    
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
        print(f"Initial matrix of evolving matrix set to shape of ( {self.m_dim}, {self.n_dim} ).")

        # Matrix after update (initialize to initial matrix)
        self.A = self.initial_matrix

        # True SVD of current update (initialize to SVD of the initial matrix)
        self.U_true, self.sigma_true, self.VH_true = np.linalg.svd(
            self.initial_matrix, full_matrices=False
        )

        # Set desired rank of truncated SVD
        if k_dim is None:
            self.k_dim = min(self.m_dim, self.n_dim)
        else:
            assert k_dim < min(self.m_dim, self.n_dim), "k must be smaller than or equal to min(m,n)."
            self.k_dim = k_dim

        # Get intial truncated SVD
        self.Uk = self.U_true[:, : self.k_dim]
        self.sigmak = self.sigma_true[: self.k_dim]
        self.VHk = self.VH_true[: self.k_dim, :]

        print(f"Initial Uk  matrix of evolving matrix set to shape of ( {np.shape(self.Uk)} ).")
        print(f"Initial Sigmak matrix of evolving matrix set to shape of ( {np.shape(self.sigmak)} ).")
        print(f"Initial VHk matrix of evolving matrix set to shape of ( {np.shape(self.VHk)} ).")

        # Initialize matrix to be appended
        self.n_batches = n_batches
        if append_matrix is None:
            self.append_matrix = np.array([])
            self.s_dim = 0
            self.step = 0
            self.U_all, self.sigma_all, self.VH_all = (
                np.array([]),
                np.array([]),
                np.array([]),
            )
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
        """Set entire matrix to appended over the course of updates and calculates SVD for final matrix
        
        Parameters
        ----------
        append_matrix : ndarray of shape (s, n)
            Matrix to be appended
        """
        self.append_matrix = append_matrix.toarray()  # ensure data is in dense format
        (self.s_dim, n_dim) = np.shape(self.append_matrix)
        self.step = int(np.ceil(self.s_dim / self.n_batches))
        assert n_dim == self.n_dim, "Number of columns must be the same for initial matrix and matrix to be appended."
        print(f"Appending matrix set to shape of ( {self.s_dim} , {self.n_dim} ).")

        self.U_all, self.Sigma_all, self.VH_all = np.linalg.svd(
            np.append(self.initial_matrix, self.append_matrix, axis=0), full_matrices=False
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
        print(f"Appending {self.n_appended} rows from appending matrix.")
        self.update_matrix = self.append_matrix[
            self.n_appended_total : self.n_appended_total + self.n_appended, :
        ]
        self.A = np.append(self.A, self.update_matrix, axis=0)

        # Update counters for update
        self.phi += 1
        self.n_appended_total += self.n_appended
        print(
            f"Appended {self.n_appended_total}/{self.s_dim} rows from appending matrix so far."
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
