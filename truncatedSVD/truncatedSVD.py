import numpy as np
import EvolvingMatrix as EM
import os

datasets = [ "CISI", "CRAN", "MED" ]
batch_splits = [ 1, 10, 12 ]
phis = [ [ 1 ], [ 1, 5, 10 ], [ 1, 6, 12 ] ]

if not os.path.exists("../cache"):
  os.mkdir("../cache")

for dataset in datasets:
  relative_errors_list = []
  residual_norms_list = []
  for batch_split, phi in zip(batch_splits, phis):
    print("Performing truncated SVD on dataset "+dataset+" using batch_split = "+str(batch_split)+".")
    if not os.path.exists("../cache/"+dataset+"_batch_split_"+str(batch_split)):
      os.mkdir("../cache/"+dataset+"_batch_split_"+str(batch_split))

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

    for ii in range(batch_split):
      Uk, Sigmak, VHk = model.evolve_matrix_zha_simon(step_dim=int(np.ceil(s_dim/batch_split)))
      if ii+1 in phi:
        model.calculate_new_svd()
        relative_errors = model.get_relative_error()
        residual_norms = model.get_residual_norm()

        print("Singular value Relative Error at phi = "+str(ii+1)+":")
        print(relative_errors)
        np.save("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/relative_errors_phi_"+str(ii+1)+".npy", relative_errors)

        print("Last singular vector Residual Norm at phi = "+str(ii+1)+":")
        print(residual_norms)
        np.save("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/residual_norms_phi_"+str(ii+1)+".npy", residual_norms)

        print()



