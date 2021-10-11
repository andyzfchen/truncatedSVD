import numpy as np

class EvolvingMatrix(object):
  def __init__(self, initial_matrix, step_dim=1, k_dim=0):
    # setting initial matrix
    self.initial_matrix = initial_matrix

    # keeping track of the initial matrix appended with rows throughout evolution
    self.A_matrix = initial_matrix

    (self.m_dim, self.n_dim) = np.shape(self.initial_matrix)
    print("Initial matrix of evolving matrix set to shape of (", self.m_dim, ",", self.n_dim, ") .")

    self.U_matrix = np.array([])
    self.Sigma_array = np.array([])
    self.V_matrix = np.array([])
    (self.U_matrix, self.Sigma_array, self.VH_matrix) = np.linalg.svd(self.initial_matrix)

    # setting truncation data
    if k_dim == 0:
      self.k_dim = self.m_dim
    else:
      self.k_dim = k_dim
    self.Uk_matrix = self.U_matrix[:self.k_dim,:]
    self.Sigmak_array = self.Sigma_array[:self.k_dim]
    self.VHk_matrix = self.VH_matrix[:self.k_dim,:]
    print("Truncated matrix of evolving matrix set to shape of (", self.k_dim, ",", self.k_dim,") .")

    # initializing appendix matrix
    self.appendix_matrix = np.array([])
    self.step_dim = step_dim

    self.s_dim = 0
    self.n_rows_appended = 0


  '''
  Initialize appendix matrix E.
  '''
  def set_appendix_matrix(self, appendix_matrix):
    self.appendix_matrix = appendix_matrix
    (self.s_dim, n_dim) = np.shape(self.appendix_matrix)
    assert n_dim == self.n_dim
    print("Appendix matrix set to shape of (", self.s_dim, ",", self.n_dim, ") .")


  '''
  First method of constructing Z = [[Uk 0] [0 Is]].
  '''
  def evolve_matrix(self, step_dim=0):
    Z = np.block(
      [[ self.Uk_matrix , np.zeros((self.k_dim, self.step_dim)) ]
       [ self.np.zeros((self.step_dim, self.n_dim)) , self.eye(self.step_dim) ]]
    )

    # default number of appended rows
    if step_dim == 0:
      step_dim = self.step_dim
    
    # checks if number of appended rows exceeds number of remaining rows in appendix matrix
    if step_dim > (self.s_dim - self.n_rows_appended):
      step_dim = self.s_dim - self.n_rows_appended

    print("Appending ", step_dim, " rows from appendix matrix and evolving truncated matrix.")

    self.A_matrix = np.append(self.A_matrix, self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:])

    (F_matrix, Theta_array, G_matrix) = np.linalg.svd(np.block(
      [[ np.diag(self.Sigmak_array) ]
       [ self.appendix_matrix[self.n_rows_appended:self.n_rows_appended+step_dim,:] ]]
    ))
    
    self.Uk_matrix = np.dot(Z, F_matrix[:self.k_dim,:])
    self.Sigmak_array = Theta_array[:k]
    self.Vk_matrix = np.dot(np.dot(self.A_matrix, self.Uk_matrix), np.diag(1/self.Sigmak_array))




