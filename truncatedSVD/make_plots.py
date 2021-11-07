import numpy as np
from plotting_helper import plot_relative_errs, plot_residual_norms, plot_stacked_relative_errs, plot_stacked_residual_norms
import os

datasets = [ "CISI", "CRAN", "MED" ]
batch_splits = [ 1, 10, 12 ]
phis = [ [ 1 ], [ 1, 5, 10 ], [ 1, 6, 12 ] ]

if not os.path.exists("../figures"):
  os.mkdir("../figures")

for dataset in datasets:
  print("Making plots for "+dataset+".")
  for batch_split, phi_list in zip(batch_splits, phis):
    # one batch update plots
    if (batch_split == 1):
      phi = 1

      relative_errors = np.load("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/relative_errors_phi_"+str(phi)+".npy")
      residual_norms = np.load("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/residual_norms_phi_"+str(phi)+".npy")

      plot_relative_errs(relative_errors, "relative_errors_"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi), title="Relative Errors for "+dataset)
      plot_residual_norms(residual_norms, "residual_norms_"+dataset+"_batch_split_"+str(batch_split)+"_phi_"+str(phi), title="Residual Norms for "+dataset)

    else:
      # 12 batch update plots
      relative_errors_list = []
      residual_norms_list = []
      for phi in phi_list:
        relative_errors = np.load("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/relative_errors_phi_"+str(phi)+".npy")
        residual_norms = np.load("../cache/"+dataset+"_batch_split_"+str(batch_split)+"/residual_norms_phi_"+str(phi)+".npy")
         
        relative_errors_list.append(relative_errors)
        residual_norms_list.append(residual_norms)

      plot_stacked_relative_errs(relative_errors_list, phi_list, "relative_errors_"+dataset+"_batch_split_"+str(batch_split), "Relative Errors for "+dataset)
      plot_stacked_residual_norms(residual_norms_list, phi_list, "residual_norms_"+dataset+"_batch_split_"+str(batch_split), "Residual Norms for "+dataset)
