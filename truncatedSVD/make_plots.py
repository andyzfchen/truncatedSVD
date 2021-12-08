import numpy as np
from plotting_helper import plot_relative_errs, plot_residual_norms, plot_stacked_relative_errs, plot_stacked_residual_norms
import os

'''
datasets = [ "CISI", "CRAN", "MED", "ML1M" ]
#datasets = [ "CISI" ]
batch_splits = [ 10 ]
phis = [ [ 1, 5, 10 ] ]
r_value = 10

'''
# debug mode
#datasets = [ "CISI", "CRAN", "MED", "ML1M" ]
datasets = [ "Reuters" ]
batch_splits = [ 1 ]
phis = [ [ 1 ] ]
r_value = 50

evolution_methods = [ "zha-simon", "bcg" ]
evolution_names = [ "$Z = [U_k, 0; 0, I_s]$", "$Z = [U_k, X_{\lambda,r}; 0, I_s]$" ]

if not os.path.exists("../figures"):
  os.mkdir("../figures")

for dataset in datasets:
  print("Making plots for "+dataset+".")
  for batch_split, phi_list in zip(batch_splits, phis):
    # one batch update plots
    if (batch_split == 1):
      phi = 1
      relative_errors_list = []
      residual_norms_list = []

      for method in evolution_methods:
        if method == "zha-simon":
          r_str = ""
        elif method == "bcg":
          r_str = "_rval_"+str(r_value)

        relative_errors = np.load("../cache/"+method+"/"+dataset+"_batch_split_"+str(batch_split)+"/relative_errors_phi_"+str(phi)+r_str+".npy")
        residual_norms = np.load("../cache/"+method+"/"+dataset+"_batch_split_"+str(batch_split)+"/residual_norms_phi_"+str(phi)+r_str+".npy")

        relative_errors_list.append(relative_errors)
        residual_norms_list.append(residual_norms)

      plot_relative_errs(relative_errors_list, evolution_names, "relative_errors_"+dataset+"_batch_split_"+str(batch_split), title=dataset)
      plot_residual_norms(residual_norms_list, evolution_names, "residual_norms_"+dataset+"_batch_split_"+str(batch_split), title=dataset)

    # multi-batch update plots
    else:
      for ii, method in enumerate(evolution_methods):
        relative_errors_list = []
        residual_norms_list = []

        if method == "zha-simon":
          r_str = ""
        elif method == "bcg":
          r_str = "_rval_"+str(r_value)

        for phi in phi_list:
          relative_errors = np.load("../cache/"+method+"/"+dataset+"_batch_split_"+str(batch_split)+"/relative_errors_phi_"+str(phi)+r_str+".npy")
          residual_norms = np.load("../cache/"+method+"/"+dataset+"_batch_split_"+str(batch_split)+"/residual_norms_phi_"+str(phi)+r_str+".npy")
           
          relative_errors_list.append(relative_errors)
          residual_norms_list.append(residual_norms)

        #plot_stacked_relative_errs(relative_errors_list, phi_list, "relative_errors_"+dataset+"_batch_split_"+str(batch_split)+"_"+method, "Relative Errors for "+dataset+" using "+evolution_names[ii])
        #plot_stacked_residual_norms(residual_norms_list, phi_list, "residual_norms_"+dataset+"_batch_split_"+str(batch_split)+"_"+method, "Residual Norms for "+dataset+" using "+evolution_names[ii])
        plot_stacked_relative_errs(relative_errors_list, phi_list, "relative_errors_"+dataset+"_batch_split_"+str(batch_split)+"_"+method, dataset+", "+evolution_names[ii])
        plot_stacked_residual_norms(residual_norms_list, phi_list, "residual_norms_"+dataset+"_batch_split_"+str(batch_split)+"_"+method, dataset+", "+evolution_names[ii])
