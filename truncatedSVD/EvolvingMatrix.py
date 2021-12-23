"""Evolving matrix for calculating truncated SVD of updated matrices.
"""

import numpy as np
import scipy.sparse.linalg
import sklearn.decomposition
import os
from metrics import proj_err, cov_err, rel_err, res_norm
from svd_update import zha_simon_update, bcg_update


class EvolvingMatrix(object):
  """Evolving matrix with periodically appended rows.
  
  This class simulates a matrix subject to the periodic addition of new rows.
  Applications with such matrices include LSI and recommender systems.
  
  Given an initial matrix, rows from a matrix to be appended are added in sequential updates.
  With each batch update, the truncated SVD of the new matrix can be calculated using 
  a variety of methods.
  
  Parameters
  ----------
  initial_matrix : ndarray of shape (s, n)
    Initial matrix
  
  step : int, default=1
    Number of batch updates
  
  k_dim : int, default=50
    Rank of truncated SVD to be calculated with each batch update
  
  Attributes
  ----------
  
  
  References
  ----------
  H. Zha and H. D. Simon, “Timely communication on updating problems in latent semantic indexing,
    ”Society for Industrial and Applied Mathematics, vol. 21, no. 2, pp. 782-791, 1999.
  
  V. Kalantzis, G. Kollias, S. Ubaru, A. N. Nikolakopoulos, L. Horesh, and K. L. Clarkson, 
    “Projection techniquesto update the truncated SVD of evolving matrices with applications,” 
    inProceedings of the 38th InternationalConference on Machine Learning, 
    M. Meila and T. Zhang, Eds.PMLR, 7 2021, pp. 5236-5246.
  """
  
  def __init__(self, initial_matrix, step=1, k_dim=None):
    # setting initial matrix
    self.initial_matrix = initial_matrix

    # keeping track of the initial matrix appended with rows throughout evolution
    self.A_matrix = initial_matrix

    (self.m_dim, self.n_dim) = np.shape(self.initial_matrix)
    print("Initial matrix of evolving matrix set to shape of (", self.m_dim, ",", self.n_dim, ") .")

    # SVD of all data
    self.U_true, self.Sigma_true, self.VH_true = np.array([]), np.array([]), np.array([])
    
    # SVD of current update
    self.U_new, self.Sigma_new, self.VH_new = np.array([]), np.array([]), np.array([])
    
    # SVD of previous update (initialize to SVD of initial matrix)
    self.U_matrix, self.Sigma_array, self.VH_matrix = np.linalg.svd(self.initial_matrix)

    # Get truncated SVD
    if k_dim is None:
      self.k_dim = self.m_dim
    else:
      self.k_dim = k_dim

    self.Uk_matrix = self.U_matrix[:, :self.k_dim]
    self.Sigmak_array = self.Sigma_array[:self.k_dim]
    self.VHk_matrix = self.VH_matrix[:self.k_dim, :]
    
    print("Initial Uk  matrix of evolving matrix set to shape of (", np.shape(self.Uk_matrix), ") .")
    print("Initial Sigmak matrix of evolving matrix set to shape of (", np.shape(self.Sigmak_array), ") .")
    print("Initial VHk matrix of evolving matrix set to shape of (", np.shape(self.VHk_matrix), ") .")

    # Initialize submatrix to be appended
    self.appendix_matrix = np.array([])
    self.step = step

    self.s_dim = 0
    self.n_rows_appended = 0
    

  def set_appendix_matrix(self, appendix_matrix):
    """Initialize matrix to append E"""
    self.appendix_matrix = appendix_matrix
    (self.s_dim, n_dim) = np.shape(self.appendix_matrix)
    assert n_dim == self.n_dim
    print("Appendix matrix set to shape of (", self.s_dim, ",", self.n_dim, ") .")

    self.U_true, self.Sigma_true, self.VH_true = np.linalg.svd(np.append(self.initial_matrix, self.appendix_matrix, axis=0))


  def evolve(self, step=None):
    """Evolve matrix"""
    
    # Default number of appended rows
    if step is None:
      step = self.step

    # Check if number of appended rows exceeds number of remaining rows in appendix matrix
    if step > (self.s_dim - self.n_rows_appended):
      step = self.s_dim - self.n_rows_appended

    # Append to current data matrix
    print(f"Appending {step} rows from appendix matrix.")
    self.A_matrix = np.append(self.A_matrix, self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step,:], axis=0)
    print(f"Appended {self.n_rows_appended}/{self.s_dim} rows from appendix matrix so far.")  
 
    # Update counter for number of rows appended
    self.n_rows_appended += step
 

  def update_svd(self):

    # Perform truncated SVD update using Zha-Simon projection algorithm
    self.Uk_matrix, self.Sigmak_array, self.VHk_matrix = zha_simon_update(self.A_matrix, 
                                                                          self.Uk_matrix, 
                                                                          self.Sigmak_array, 
                                                                          self.VHk_matrix, 
                                                                          self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim, :])
    
    return None
  

  def evolve_matrix_brute_force(self):
    """Calculate optimal rank-k approximation of A using brute force"""
    return self.Uk_matrix, self.Sigmak_array, self.VHk_matrix


  def evolve_matrix_zha_simon(self, step_dim=None):
    """Construct Z using Zha-Simon algorithm.
    
    Z = [[U_k 0] [0 I_s]]
    """
    print("Using Zha-Simon method to updated truncated SVD of evolved matrix.")

    # default number of appended rows
    if step_dim is None:
      step_dim = self.step_dim

    # checks if number of appended rows exceeds number of remaining rows in appendix matrix
    if step_dim > (self.s_dim - self.n_rows_appended):
      step_dim = self.s_dim - self.n_rows_appended

    # # kSVD of ZH*A        
    # # TODO: implement unrestarted Lanczos method on ZH*A*AH*Z
    # print("Performing kSVD on ZH*A.")
    # (F_matrix, Theta_array, G_matrix) = np.linalg.svd(np.block(
    #   [[ np.dot(np.diag(self.Sigmak_array), self.VHk_matrix) ],
    #    [ self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:] ]]
    # ), full_matrices=False)

    # Append to current data matrix
    print("Appending ", step_dim, " rows from appendix matrix and evolving truncated matrix.")
    self.A_matrix = np.append(self.A_matrix, self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:], axis=0)
    print("Appended ", self.n_rows_appended, " of ", self.s_dim, " rows from appendix matrix so far.")
    
    # Perform truncated SVD update using Zha-Simon projection algorithm
    self.Uk_matrix, self.Sigmak_array, self.VHk_matrix = zha_simon_update(self.A_matrix, 
                                                                          self.Uk_matrix, 
                                                                          self.Sigmak_array, 
                                                                          self.VHk_matrix, 
                                                                          self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim, :])

    # Update counter for number of rows appended
    self.n_rows_appended += step_dim

    return self.Uk_matrix, self.Sigmak_array, self.VHk_matrix


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
    E_matrix = self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:]
    svd = sklearn.decomposition.TruncatedSVD(n_components=5, n_iter=20)
    svd.fit(np.append(self.A_matrix, self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:], axis=0))
    #lambda_value = 1.01 * self.Sigmak_array[0]**2   # lambda_value should be >= first singular value
    lambda_value = 1.01 * svd.singular_values_[0]**2   # lambda_value should be >= first singular value
    LHS_matrix = -( np.dot(self.A_matrix, self.A_matrix.T) - lambda_value*np.eye(self.m_dim+self.n_rows_appended) )
    RHS_matrix = np.dot( ( np.eye(self.m_dim+self.n_rows_appended) - np.dot(self.Uk_matrix, self.Uk_matrix.T) ), np.dot(self.A_matrix, E_matrix.T) )

    '''
    X_matrix = np.zeros((self.m_dim+self.n_rows_appended, step_dim))
    # TODO: currently using CG column by column; need to implement block CG instead
    for ii in range(step_dim):
      if (ii+1)%50 == 0:
        print("Step "+str(ii+1)+" of "+str(step_dim)+".")
      X_matrix[:,ii] = scipy.sparse.linalg.cg(LHS_matrix, RHS_matrix[:,ii])[0]
    print("Inverting matrix for X matrix.")
    '''
    X_matrix = np.dot(np.linalg.inv(LHS_matrix), RHS_matrix)



    # rSVD of X
    print("Performing randomized rSVD on X for X_lambda,r matrix.")
    X_matrix = np.dot(X_matrix, np.random.normal(size=(step_dim, 2*r_dim)))
    (Xlambdar_matrix, Slambdar_array, Ylambdar_matrix) = np.linalg.svd(X_matrix, full_matrices=False)

    # Z matrix
    Z_matrix = np.block(
      [[ self.Uk_matrix , Xlambdar_matrix[:,:r_dim], np.zeros((self.m_dim+self.n_rows_appended, step_dim)) ],
       [ np.zeros((step_dim, self.k_dim+r_dim)) , np.eye(step_dim) ]]
    )

    # kSVD of ZH*A    # TODO: implement unrestarted Lanczos method on ZH*A*AH*Z
    print("Performing kSVD on ZH*A.")
    ZHA = np.block(
      [[ np.dot(np.diag(self.Sigmak_array), self.VHk_matrix) ],
       [ np.dot(Xlambdar_matrix[:,:r_dim].T, self.A_matrix) ],
       [ self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:] ]])
    '''
    ZHA = np.block([ np.dot(self.VHk_matrix.T, np.diag(self.Sigmak_array)), np.dot(self.A_matrix.T, Xlambdar_matrix[:,:r_dim]), self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:].T ]).T
    '''
    (F_matrix, Theta_array, G_matrix) = np.linalg.svd(ZHA, full_matrices=False)
    
    print("Appending ", step_dim, " rows from appendix matrix and evolving truncated matrix.")

    self.A_matrix = np.append(self.A_matrix, self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:], axis=0)

    self.n_rows_appended += step_dim

    print("Appended ", self.n_rows_appended, " of ", self.s_dim, " rows from appendix matrix so far.")
    
    # recalculation of Uk, Sigmak, and VHk
    print("Recalculating U_k, Sigma_k, and VH_k.")
    self.Uk_matrix = np.dot(Z_matrix, F_matrix[:,:self.k_dim])
    self.Sigmak_array = Theta_array[:self.k_dim]
    self.VHk_matrix = np.dot(np.dot(self.A_matrix.T, self.Uk_matrix), np.diag(1/self.Sigmak_array)).T

    print()

    return self.Uk_matrix, self.Sigmak_array, self.VHk_matrix


  def calculate_new_svd(self, evolution_method, dataset, batch_split, phi):
    """Calculate SVD components of the updated A matrix."""
    print("Calculating current A matrix of size ", np.shape(self.A_matrix))

    # Folder to save U,S,V arrays
    dirname = "../cache/"+evolution_method+"/"+dataset+"_batch_split_"+str(batch_split)
    
    if os.path.exists(f"{dirname}/U_matrix_phi_{str(phi+1)}.npy"):
      print("Loading from cache")
      self.U_new = np.load(f"{dirname}/U_matrix_phi_{str(phi+1)}.npy")
      self.Sigma_new = np.load(f"{dirname}/Sigma_array_phi_{str(phi+1)}.npy", self.Sigma_new)
      self.VH_new = np.load(f"{dirname}/VH_matrix_phi_{str(phi+1)}.npy", self.VH_new)
    else:
      self.U_new, self.Sigma_new, self.VH_new = np.linalg.svd(self.A_matrix)
      
      np.save(f"{dirname}/U_matrix_phi_{str(phi+1)}.npy", self.U_new)
      np.save(f"{dirname}/Sigma_array_phi_{str(phi+1)}.npy", self.Sigma_new)
      np.save(f"{dirname}/VH_matrix_phi_{str(phi+1)}.npy", self.VH_new)


  def get_relative_error(self, sv_idx=None):
    """Return relative error of n-th singular value"""
    if sv_idx is None:
      sv_idx = np.arange(self.k_dim)

    return rel_err(self.Sigma_new[sv_idx], self.Sigmak_array[sv_idx]) 


  def get_residual_norm(self, sv_idx=None):
    """Return residual norm of n-th singular vector"""
    if sv_idx is None:
      sv_idx = np.arange(self.k_dim)

    return res_norm(self.A_matrix, self.Uk_matrix[:, sv_idx], self.VHk_matrix[sv_idx, :].T, self.Sigmak_array[sv_idx])


  def get_covariance_error(self):
    """Return covariance error"""
    return cov_err(self.A_matrix, self.U_new.dot(np.diag(self.Sigma_new).dot(self.VH_new)))


  def get_projection_error(self):
    """Return projection error"""
    # TODO: calculate projection Ahat
