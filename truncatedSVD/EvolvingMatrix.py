import numpy as np
import scipy.sparse.linalg
import os

class EvolvingMatrix(object):
  def __init__(self, initial_matrix, step_dim=1, k_dim=0):
    # setting initial matrix
    self.initial_matrix = initial_matrix

    # keeping track of the initial matrix appended with rows throughout evolution
    self.A_matrix = initial_matrix

    (self.m_dim, self.n_dim) = np.shape(self.initial_matrix)
    print("Initial matrix of evolving matrix set to shape of (", self.m_dim, ",", self.n_dim, ") .")

    self.U_true = np.array([])
    self.Sigma_true = np.array([])
    self.VH_true = np.array([])
    self.U_new = np.array([])
    self.Sigma_new = np.array([])
    self.VH_new = np.array([])
    self.U_matrix = np.array([])
    self.Sigma_array = np.array([])
    self.VH_matrix = np.array([])
    (self.U_matrix, self.Sigma_array, self.VH_matrix) = np.linalg.svd(self.initial_matrix)

    # setting truncation data
    if k_dim == 0:
      self.k_dim = self.m_dim
    else:
      self.k_dim = k_dim
    self.Uk_matrix = self.U_matrix[:,:self.k_dim]
    self.Sigmak_array = self.Sigma_array[:self.k_dim]
    self.VHk_matrix = self.VH_matrix[:self.k_dim,:]
    print("Initial Uk  matrix of evolving matrix set to shape of (", np.shape(self.Uk_matrix), ") .")
    print("Initial Sigmak matrix of evolving matrix set to shape of (", np.shape(self.Sigmak_array), ") .")
    print("Initial VHk matrix of evolving matrix set to shape of (", np.shape(self.VHk_matrix), ") .")

    # initializing appendix matrix
    self.appendix_matrix = np.array([])
    self.step_dim = step_dim

    self.s_dim = 0
    self.n_rows_appended = 0

    print()


  '''
  Initialize appendix matrix E.
  '''
  def set_appendix_matrix(self, appendix_matrix):
    self.appendix_matrix = appendix_matrix
    (self.s_dim, n_dim) = np.shape(self.appendix_matrix)
    assert n_dim == self.n_dim
    print("Appendix matrix set to shape of (", self.s_dim, ",", self.n_dim, ") .")

    self.U_true, self.Sigma_true, self.VH_true = np.linalg.svd(np.append(self.initial_matrix, self.appendix_matrix, axis=0))

    print()


  '''
  First method of constructing Z = [[U_k 0] [0 I_s]].
  '''
  def evolve_matrix_zha_simon(self, step_dim=None):
    print("Using Zha Simon method to evolve matrix.")

    # default number of appended rows
    if step_dim is None:
      step_dim = self.step_dim

    # checks if number of appended rows exceeds number of remaining rows in appendix matrix
    if step_dim > (self.s_dim - self.n_rows_appended):
      step_dim = self.s_dim - self.n_rows_appended

    # Z matrix
    Z_matrix = np.block(
      [[ self.Uk_matrix , np.zeros((self.m_dim+self.n_rows_appended, step_dim)) ],
       [ np.zeros((step_dim, self.k_dim)) , np.eye(step_dim) ]]
    )

    # kSVD of ZH*A        # TODO: implement unrestarted Lanczos method on ZH*A*AH*Z
    print("Performing kSVD on ZH*A.")
    (F_matrix, Theta_array, G_matrix) = np.linalg.svd(np.block(
      [[ np.dot(np.diag(self.Sigmak_array), self.VHk_matrix) ],
       [ self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:] ]]
    ), full_matrices=False)
    
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


  '''
  Second method of constructing Z = [[U_k, X_lambda,r 0] [0 I_s]].
  '''
  def evolve_matrix_deflated_bcg(self, step_dim=None, r_dim=None):
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
    lambda_value = 1.01 * self.Sigmak_array[0]**2   # lambda_value should be >= first singular value
    LHS_matrix = -( np.dot(self.A_matrix, self.A_matrix.T) - lambda_value*np.eye(self.m_dim+self.n_rows_appended) )
    RHS_matrix = np.dot( ( np.eye(self.m_dim+self.n_rows_appended) - np.dot(self.Uk_matrix, self.Uk_matrix.T) ), np.dot(self.A_matrix, E_matrix.T) )

    X_matrix = np.zeros((self.m_dim+self.n_rows_appended, step_dim))
    '''
    # TODO: currently using CG column by column; need to implement block CG instead
    for ii in range(step_dim):
      print("Step "+str(ii+1)+" of "+str(step_dim)+".")
      X_matrix[:,ii] = scipy.sparse.linalg.cg(LHS_matrix, RHS_matrix[:,ii])[0]
    '''
    print("Inverting matrix for X matrix.")
    X_matrix = np.dot(np.linalg.inv(LHS_matrix), RHS_matrix)



    # rSVD of X
    print("Performing rSVD on X for X_lambda,r matrix.")
    (Xlambdar_matrix, Slambdar_array, Ylambdar_matrix) = np.linalg.svd(X_matrix, full_matrices=False)

    # Z matrix
    Z_matrix = np.block(
      [[ self.Uk_matrix , Xlambdar_matrix[:,:r_dim], np.zeros((self.m_dim+self.n_rows_appended, step_dim)) ],
       [ np.zeros((step_dim, self.k_dim+r_dim)) , np.eye(step_dim) ]]
    )

    # kSVD of ZH*A    # TODO: implement unrestarted Lanczos method on ZH*A*AH*Z
    print("Performing kSVD on ZH*A.")
    (F_matrix, Theta_array, G_matrix) = np.linalg.svd(np.block(
      [[ np.dot(np.diag(self.Sigmak_array), self.VHk_matrix) ],
       [ np.dot(Xlambdar_matrix[:,:r_dim].T, self.A_matrix) ],
       [ self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:] ]]
    ), full_matrices=False)
    
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


  '''
  Calculate the SVD components of the A matrix so far.
  '''
  def calculate_new_svd(self, dataset, batch_split, phi):
    print("Calculating current A matrix of size ", np.shape(self.A_matrix))

    if os.path.exists("../cache/"+dataset+"/"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi+1)+"_Sigma_array.npy"):
      self.U_new = np.load("../cache/"+dataset+"/"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi+1)+"_U_matrix.npy")
      self.Sigma_new = np.load("../cache/"+dataset+"/"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi+1)+"_Sigma_array.npy")
      self.VH_new = np.load("../cache/"+dataset+"/"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi+1)+"_VH_matrix.npy")
    else:
      self.U_new, self.Sigma_new, self.VH_new = np.linalg.svd(self.A_matrix)

      np.save("../cache/"+dataset+"/"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi+1)+"_U_matrix.npy", self.U_new)
      np.save("../cache/"+dataset+"/"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi+1)+"_Sigma_array.npy", self.Sigma_new)
      np.save("../cache/"+dataset+"/"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi+1)+"_VH_matrix.npy", self.VH_new)
    return


  '''
  Return the relative error of the nth (sv_idx) singular value.
  '''
  def get_relative_error(self, sv_idx=None):
    if sv_idx is None:
      sv_idx = np.arange(self.k_dim)
  
    return np.abs(self.Sigmak_array[sv_idx] - self.Sigma_new[sv_idx]) / self.Sigma_new[sv_idx]


  '''
  Return the residual norm of the nth (sv_idx) singular vector.
  '''
  def get_residual_norm(self, sv_idx=None):
    if sv_idx is None:
      sv_idx = np.arange(self.k_dim)

    return np.linalg.norm(np.dot(self.A_matrix, self.VHk_matrix[sv_idx,:].T) - self.Uk_matrix[:,sv_idx]*self.Sigmak_array[sv_idx],axis=0) / self.Sigmak_array[sv_idx]


