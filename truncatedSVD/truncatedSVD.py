import numpy as np
import EvolvingMatrix as EM

datasets = [ "CISI", "CRAN", "MED" ]

for dataset in datasets:
  print("Performing truncated SVD on dataset "+dataset+".")
  A_full = np.load("../datasets/"+dataset+"/"+dataset+".npy")
  m_percent = 0.50

  (m_dim_full, n_dim) = np.shape(A_full)
  m_dim = int(np.ceil(m_dim_full*m_percent))
  s_dim = int(np.floor(m_dim_full*(1-m_percent)))
  k_dim = 50

  B = A_full[:m_dim,:]
  E = A_full[m_dim:,:]

  model = EM.EvolvingMatrix(B, k_dim=k_dim)
  model.set_appendix_matrix(E)

  Uk, Sigmak, VHk = model.evolve_matrix(step_dim=s_dim)

  print("Last singular vector Residual Norm:")
  print(model.get_residual_norm())
  print("Last singular value Relative Error:")
  print(model.get_relative_error())

  print()
