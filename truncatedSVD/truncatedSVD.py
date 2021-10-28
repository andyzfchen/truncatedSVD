import numpy as np
import EvolvingMatrix as EM

datasets = [ "CISI", "CRAN", "MED" ]
#phis = [ -1, 1, 12 ]
phis = [ -1 ]

for phi in phis:
  for dataset in datasets:
    print("Performing truncated SVD on dataset "+dataset+" using phi = "+str(phi)+".")
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

    if phi == -1:
      for ii in range(s_dim):
        Uk, Sigmak, VHk = model.evolve_matrix(step_dim=1)
    else:
      for ii in range(phi):
        Uk, Sigmak, VHk = model.evolve_matrix(step_dim=int(np.ceil(s_dim/phi)))

    relative_errors = model.get_relative_error()
    residual_norms = model.get_residual_norm()

    print("Last singular value Relative Error:")
    print(relative_errors)
    np.save("../cache/relative_errors_"+dataset+"_phi_"+str(phi)+".npy", relative_errors)
    print("Last singular vector Residual Norm:")
    print(residual_norms)
    np.save("../cache/residual_norms_"+dataset+"_phi_"+str(phi)+".npy", residual_norms)

    print()
