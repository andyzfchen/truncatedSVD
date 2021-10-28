import numpy as np

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
  First method of constructing Z = [[Uk 0] [0 Is]].
  '''
  def evolve_matrix(self, step_dim=None):
    # default number of appended rows
    if step_dim is None:
      step_dim = self.step_dim

    # checks if number of appended rows exceeds number of remaining rows in appendix matrix
    if step_dim > (self.s_dim - self.n_rows_appended):
      step_dim = self.s_dim - self.n_rows_appended

    Z = np.block(
      [[ self.Uk_matrix , np.zeros((self.m_dim+self.n_rows_appended, step_dim)) ],
       [ np.zeros((step_dim, self.k_dim)) , np.eye(step_dim) ]]
    )

    print("Appending ", step_dim, " rows from appendix matrix and evolving truncated matrix.")

    self.A_matrix = np.append(self.A_matrix, self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:], axis=0)

    (F_matrix, Theta_array, G_matrix) = np.linalg.svd(np.block(
      [[ np.dot(np.diag(self.Sigmak_array), self.VHk_matrix) ],
       [ self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:] ]]
    ))
    
    self.n_rows_appended += step_dim

    print("Appended ", self.n_rows_appended, " of ", self.s_dim, " rows from appendix matrix so far.")
    

    self.Uk_matrix = np.dot(Z, F_matrix[:,:self.k_dim])
    self.Sigmak_array = Theta_array[:self.k_dim]

    self.VHk_matrix = np.dot(np.dot(self.A_matrix.T, self.Uk_matrix), np.diag(1/self.Sigmak_array)).T

    print()

    return self.Uk_matrix, self.Sigmak_array, self.VHk_matrix


  '''
  Return the relative error of the nth (sv_idx) singular value.
  '''
  def get_relative_error(self, sv_idx=None):
    if sv_idx is None:
      sv_idx = np.arange(self.k_dim)

    return np.abs(self.Sigmak_array[sv_idx] - self.Sigma_true[sv_idx]) / self.Sigma_true[sv_idx]


  '''
  Return the residual norm of the nth (sv_idx) singular vector.
  '''
  def get_residual_norm(self, sv_idx=None):
    if sv_idx is None:
      sv_idx = np.arange(self.k_dim)

    return np.linalg.norm(np.dot(self.A_matrix, self.VHk_matrix[sv_idx,:].T) - self.Uk_matrix[:,sv_idx]*self.Sigmak_array[sv_idx],axis=0) / self.Sigmak_array[sv_idx]


