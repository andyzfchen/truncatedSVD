import os
import numpy as np
from plotting_helper import (
    plot_relative_errors,
    plot_residual_norms,
    plot_stacked_relative_errors,
    plot_stacked_residual_norms,
)


datasets = ["CISI", "CRAN", "MED", "ML1M", "Reuters"]
batch_splits = [10]
phis = [[1, 5, 10]]
r_value = 10
update_methods = ["zha-simon", "bcg"]
update_names = ["$Z = [U_k, 0; 0, I_s]$", "$Z = [U_k, X_{\lambda,r}; 0, I_s]$"]


# debug mode
datasets = ["CISI", "CRAN", "MED"]
batch_splits = [10]
phis = [[1, 5, 10]]
r_value = 10
# evolution_methods = [ "bcg" ]
# evolution_names = [ "$Z = [U_k, X_{\lambda,r}; 0, I_s]$" ]
update_methods = ["zha-simon", "bcg"]
update_names = [
    "$Z = [U_k, 0; 0, I_s]$",
    "$Z = [U_k, X_{\lambda,r}; 0, I_s]$",
]


if not os.path.exists("../figures"):
    os.mkdir("../figures")

for dataset in datasets:
    print(f"Making plots for {dataset}.")
    for batch_split, phi_list in zip(batch_splits, phis):

        # Single batch update plots
        if batch_split == 1:
            relative_errors_list = []
            residual_norms_list = []

            for method in update_methods:
                if method == "zha-simon":
                    r_str = ""
                elif method == "bcg":
                    r_str = "_rval_" + str(r_value)
                
                folder = f"../cache/{method}/{dataset}_batch_split_{str(batch_split)}"

                relative_errors = np.load(f"{folder}/relative_errors_phi_1{r_str}.npy")
                residual_norms = np.load(f"{folder}/residual_norms_phi_1{r_str}.npy")

                relative_errors_list.append(relative_errors)
                residual_norms_list.append(residual_norms)

                plot_relative_errors(
                    relative_errors_list,
                    update_names,
                    f"relative_errors_{dataset}_batch_split_{str(batch_split)}",
                    title=dataset,
                )
                plot_residual_norms(
                    residual_norms_list,
                    update_names,
                    f"residual_norms_{dataset}_batch_split_{str(batch_split)}",
                    title=dataset,
                )

        # Multi-batch update plots
        else:
            for ii, method in enumerate(update_methods):
                relative_errors_list = []
                residual_norms_list = []

                if method == "zha-simon":
                    r_str = ""
                elif method == "bcg":
                    r_str = "_rval_" + str(r_value)
                
                folder = f"../cache/{method}/{dataset}_batch_split_{str(batch_split)}"

                for phi in phi_list:
                    relative_errors = np.load(f"{folder}/relative_errors_phi_{str(phi)}{r_str}.npy")
                    residual_norms = np.load(f"{folder}/residual_norms_phi_{str(phi)}{r_str}.npy")

                    relative_errors_list.append(relative_errors)
                    residual_norms_list.append(residual_norms)

                title = f"{dataset}, {update_names[ii]}"
                # plot_stacked_relative_errs(relative_errors_list, phi_list, "relative_errors_"+dataset+"_batch_split_"+str(batch_split)+"_"+method, "Relative Errors for "+dataset+" using "+evolution_names[ii])
                # plot_stacked_residual_norms(residual_norms_list, phi_list, "residual_norms_"+dataset+"_batch_split_"+str(batch_split)+"_"+method, "Residual Norms for "+dataset+" using "+evolution_names[ii])
                plot_stacked_relative_errors(
                    relative_errors_list,
                    phi_list,
                    f"relative_errors_{dataset}_batch_split_{str(batch_split)}_{method}",
                    title=title,
                )
                plot_stacked_residual_norms(
                    residual_norms_list,
                    phi_list,
                    f"residual_norms_{dataset}_batch_split_{str(batch_split)}_{method}",
                    title=title,
                )
