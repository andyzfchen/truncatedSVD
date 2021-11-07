import numpy as np
from plotting_helper import plot_relative_errs, plot_residual_norms, plot_stacked_relative_errs, plot_stacked_residual_norms
import os

#datasets = [ "CISI", "CRAN", "MED" ]
datasets = [ "MED" ]
phis = [ 1, 5, 10 ]

for dataset in datasets:
  # one batch update plots
  batch_split = 1
  phi = 1

  relative_errors = np.load("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/relative_errors_phi_"+str(phi)+".npy")
  residual_norms = np.load("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/residual_norms_phi_"+str(phi)+".npy")

  plot_relative_errs(relative_errors, "relative_errors_"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi), title="Relative Errors for "+dataset+", batch_split = "+str(batch_split)+", $\phi$ = "+str(phi))
  plot_residual_norms(residual_norms, "residual_norms_"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi), title="Residual Norms for "+dataset+", batch_split = "+str(batch_split)+", $\phi$ = "+str(phi))

  # 12 batch update plots
  batch_split = 10
  relative_errors_list = []
  residual_norms_list = []
  for phi in phis:
    relative_errors = np.load("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/relative_errors_phi_"+str(phi)+".npy")
    residual_norms = np.load("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/residual_norms_phi_"+str(phi)+".npy")
     
    relative_errors_list.append(relative_errors)
    residual_norms_list.append(residual_norms)

  plot_stacked_relative_errs(relative_errors_list[0], relative_errors_list[1], relative_errors_list[2], "relative_errors_"+dataset+"_batch_split_"+str(batch_split), "Relative Errors for "+dataset)
  plot_stacked_residual_norms(residual_norms_list[0], residual_norms_list[1], residual_norms_list[2], "residual_norms_"+dataset+"_batch_split_"+str(batch_split), "Residual Norms for "+dataset)
