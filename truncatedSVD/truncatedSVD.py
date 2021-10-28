import numpy as np
import EvolvingMatrix as EM
from plotting_helper import plot_relative_errs, plot_residual_norms, plot_stacked_relative_errs, plot_stacked_residual_norms
import os

datasets = [ "CISI", "CRAN", "MED" ]
phis = [ 1, 6, 12 ]

for dataset in datasets:
  relative_errors_list = []
  residual_norms_list = []
  for phi in phis:
    if os.path.exists("../cache/relative_errors_"+dataset+"_phi_"+str(phi)+".npy"):
      print("Loading relative errors and residual norms for dataset "+dataset+" using phi = "+str(phi)+".")
      relative_errors = np.load("../cache/relative_errors_"+dataset+"_phi_"+str(phi)+".npy")
      residual_norms = np.load("../cache/residual_norms_"+dataset+"_phi_"+str(phi)+".npy")
    else:
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
    plot_relative_errs(relative_errors, "relative_errors_"+dataset+"_phi_"+str(phi), "Relative Errors for "+dataset+", $\phi$ = "+str(phi))


    print("Last singular vector Residual Norm:")
    print(residual_norms)
    np.save("../cache/residual_norms_"+dataset+"_phi_"+str(phi)+".npy", residual_norms)
    plot_residual_norms(residual_norms, "residual_norms_"+dataset+"_phi_"+str(phi), "Residual Norms for "+dataset+", $\phi$ = "+str(phi))

    relative_errors_list.append(relative_errors)
    residual_norms_list.append(residual_norms)

    print()

  plot_stacked_relative_errs(relative_errors_list[0], relative_errors_list[1], relative_errors_list[2], "relative_errors_"+dataset, "Relative Errors for "+dataset)
  plot_stacked_residual_norms(residual_norms_list[0], residual_norms_list[1], residual_norms_list[2], "residual_norms_"+dataset, "Residual Norms for "+dataset)




