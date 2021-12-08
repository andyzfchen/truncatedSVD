import numpy as np
import os
import matplotlib.pyplot as plt

n_value = 100
r_value = 10
datasets = [ "CISI", "CRAN", "MED", "ML1M" ]
batch_split = 10
phis =  [ 1, 5, 10 ]
evolution_methods = [ "zha-simon", "bcg" ]
error_types = [ "relative_errors", "residual_norms" ]

#if not os.path.exists("../cache"):
#  os.mkdir("../cache")

#Sigma_array_list = []

for dataset in datasets:
  for error_type in error_types:
    errors_list = []

    for ii, phi in enumerate(phis):
      errors_list.append([])

      for evolution_method in evolution_methods:
        if evolution_method == "zha-simon":
          r_str = ""
        elif evolution_method == "bcg":
          r_str = "_rval_"+str(r_value)

        errors_list[ii].append(np.load("../cache/"+evolution_method+"/"+dataset+"_batch_split_"+str(batch_split)+"/"+error_type+"_phi_"+str(phi)+r_str+".npy")[1:])


    for ii, phi in enumerate(phis):
      string = ""
      for jj, evolution_method in enumerate(evolution_methods):
        string += " & $ %.3f \\pm %.3f $" % (np.mean(np.log10(errors_list[ii][jj]/errors_list[0][jj])), np.std(np.log10(errors_list[ii][jj]/errors_list[0][jj])))
      print(dataset, error_type, "phi=", phi, string, "\\\\")
  

