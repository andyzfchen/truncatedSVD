import os
import numpy as np
from plotting_helper import (
    plot_relative_errors,
    plot_residual_norms,
    plot_stacked_relative_errors,
    plot_stacked_residual_norms,
    plot_errors,
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
update_methods = [
    "zha-simon", 
    "bcg",
    # "brute-force"
    # "naive",
    # "fd"
    ]
update_names = [
    # "$Z = [U_k, 0; 0, I_s]$",
    # "$Z = [U_k, X_{\lambda,r}; 0, I_s]$",
    # "Brute force",
    # "Naive"
    "FD"
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
            # Initialize lists to store covariance and projection errors
            covariance_errors_list = []
            projection_errors_list = []
            update_method_list = []
            
            for ii, method in enumerate(update_methods):
                # Initilize lists to store metrics
                relative_errors_list = []
                residual_norms_list = []

                if method == "zha-simon" or method == "fd":
                    r_str = ""
                elif method == "brute-force" or method == "naive":
                    continue
                elif method == "bcg":
                    r_str = "_rval_" + str(r_value)

                folder = f"../cache/{method}/{dataset}_batch_split_{str(batch_split)}"

                # Append metrics to list for each phi
                for phi in phi_list:
                    rel_err = np.load(f"{folder}/relative_errors_phi_{str(phi)}{r_str}.npy")
                    res_norm = np.load(f"{folder}/residual_norms_phi_{str(phi)}{r_str}.npy")
                    cov_err = np.load(f"{folder}/covariance_errors_phi_{str(phi)}{r_str}.npy")
                    # proj_err = np.load(f"{folder}/projection_errors_phi_{str(phi)}{r_str}.npy") 

                    relative_errors_list.append(rel_err)
                    residual_norms_list.append(res_norm)
                    covariance_errors_list.append(cov_err)
                    # projection_errors_list.append(proj_err)
                    update_method_list.append(method)

                title = f"{dataset}, {update_names[ii]}"
            
                # Plot metrics for each update method
                plot_stacked_relative_errors(
                    relative_errors_list,
                    phi_list,
                    f"relative_errors_{dataset}_batch_split_{str(batch_split)}_{method}",
                    title=title
                )

                plot_stacked_residual_norms(
                    residual_norms_list,
                    phi_list,
                    f"residual_norms_{dataset}_batch_split_{str(batch_split)}_{method}",
                    title=title
                )
            
            # Plot covariance and projection errors for each update method
            plot_errors(
                covariance_errors_list, 
                projection_errors_list, 
                phi_list,
                update_methods, 
                f"proj_cov_err_{dataset}_batch_split_{str(batch_split)}", 
                title=dataset
            )
