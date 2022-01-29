"""Script to run experiments for updating the truncated SVD of evolving matrices.
"""

import sys
from os import mkdir
from os.path import normpath, exists, join
import json
import numpy as np
import EvolvingMatrix as EM
from test_plotter import plot_stacked_residual_norms, plot_stacked_relative_errors


def perform_updates(
    dataset,
    n_batches,
    phi,
    model,
    method,
    update_method,
    r_str,
    save_dir,
    res_norms_list,
    rel_errs_list,
    make_plots=False,
    **kwargs,
):
    """Perform updates for specified number of batches using update method."""

    print(f"Update method: {method}")
    for ii in range(n_batches):
        print(f"Batch {str(ii+1)}/{str(n_batches)}.")

        # Evolve matrix by appending new rows
        model.evolve()
        if not kwargs:
            update_method()
        else:
            update_method(**kwargs)

        # Save results if batch number specified
        if model.phi in phi or ii == n_batches - 1:

            # Calculate true SVD for this batch
            model.calculate_true_svd(method, dataset)

            # Caluclate metrics
            if method == "frequent-directions":
                print(f"INSIDE FREQUENT DIRECTIONS: {model.freq_dir.ell}\n")
                model.save_metrics(
                    save_dir, print_metrics=True, A_idx=model.freq_dir.ell, r_str=r_str
                )
            else:
                model.save_metrics(save_dir, print_metrics=True, r_str=r_str)

            # gather results for plotting
            if make_plots:
                rel_err = model.get_relative_error(sv_idx=None)
                if method == "frequent-directions":
                    res_norm = model.get_residual_norm(
                        sv_idx=None, A_idx=model.freq_dir.ell
                    )
                else:
                    res_norm = model.get_residual_norm(sv_idx=None)

                res_norms_list.append(res_norm)
                rel_errs_list.append(rel_err)

        print()


# Check for correct number of arguments
if len(sys.argv) != 3:
    print(
        "Usage: python run_tests.py <path to test json file> <path to folder to hold cache>"
    )
    exit()

if not exists(sys.argv[1]):
    print(
        "Usage: python run_tests.py <path to test json file> <path to folder to hold cache>\nPath to test file does not exist"
    )
    exit()
if not exists(sys.argv[2]):
    print(
        "Usage: python run_tests.py <path to test json file> <path to folder to hold cache>\nPath to parent directory for cache does not exist"
    )
    exit()

# Open JSON file for experiment specification
f = open(sys.argv[1])

try:
    test_spec = json.load(f)
except ValueError as err:
    print("Tests file is not valid json, please double check syntax")
    exit()

# Create cache path to save results
cache_path = normpath(join(sys.argv[2], "cache"))

if not exists(cache_path):
    mkdir(normpath(cache_path))

# Run tests for each method
for test in test_spec["tests"]:

    method = test["method"]

    # Create directory to store results for method
    if not exists(normpath(join(cache_path, method))):
        mkdir(normpath(join(cache_path, method)))

    for dataset in test["datasets"]:

        # Create directory to store results for method and dataset
        if not exists(normpath(join(cache_path, method, dataset))):
            mkdir(normpath(join(cache_path, method, dataset)))

        data = np.load(test_spec["dataset_info"][dataset])

        if data.shape[0] < data.shape[1]:
            data = data.T

        for n_batches in test["n_batches"]:
            batch_phis = [x for x in test["phis_to_plot"] if x <= n_batches]
            if n_batches not in batch_phis:
                batch_phis.append(n_batches)

            # Calculate data split indices
            m_percent = test["m_percent"]
            (m_dim_full, n_dim) = np.shape(data)
            print(f"Data is of shape ({m_dim_full}, {n_dim})")

            m_dim = int(np.ceil(m_dim_full * m_percent))
            s_dim = int(np.floor(m_dim_full * (1 - m_percent)))

            # Split into initial matrix and matrix to be appended
            B = data[:m_dim, :]
            E = data[m_dim:, :]

            for k in test["k_dims"]:

                print(
                    f"Performing truncated SVD on dataset {dataset} using batch_split = {str(n_batches)} and k_dims = {str(k)}."
                )

                # Create directory to save data for this batch split and k
                save_dir = join(
                    cache_path,
                    method,
                    dataset,
                    f"{dataset}_batch_split_{str(n_batches)}_k_dims_{str(k)}",
                )
                if not exists(normpath(save_dir)):
                    mkdir(normpath(save_dir))

                res_norms_list = []
                rel_errs_list = []
                if method == "frequent-directions":
                    model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k)
                    model.set_append_matrix(E)
                    print()
                    perform_updates(
                        dataset,
                        n_batches,
                        batch_phis,
                        model,
                        method,
                        model.update_svd_fd,
                        "",
                        save_dir,
                        res_norms_list,
                        rel_errs_list,
                        make_plots=test["make_plots"],
                    )
                    if test["make_plots"]:
                        fig_dir = save_dir + "/figures"
                        if not exists(normpath(fig_dir)):
                            mkdir(normpath(fig_dir))
                        plot_stacked_relative_errors(rel_errs_list, batch_phis, fig_dir)
                        plot_stacked_residual_norms(res_norms_list, batch_phis, fig_dir)
                elif method == "zha-simon":
                    # Initialize EM object with initial matrix and matrix to be appended and set desired rank of truncated SVD
                    model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k)
                    model.set_append_matrix(E)
                    print()
                    perform_updates(
                        dataset,
                        n_batches,
                        batch_phis,
                        model,
                        method,
                        model.update_svd_zha_simon,
                        "",
                        save_dir,
                        res_norms_list,
                        rel_errs_list,
                        make_plots=test["make_plots"],
                    )
                    if test["make_plots"]:
                        fig_dir = save_dir + "/figures"
                        if not exists(normpath(fig_dir)):
                            mkdir(normpath(fig_dir))
                        plot_stacked_relative_errors(rel_errs_list, batch_phis, fig_dir)
                        plot_stacked_residual_norms(res_norms_list, batch_phis, fig_dir)
                elif method == "bcg":
                    for r in test["r_values"]:
                        for run_num in range(test["num_runs"]):
                            res_norms_list = []
                            rel_errs_list = []
                            save_dir_run = normpath(
                                join(save_dir, f"run_{str(run_num)}")
                            )
                            if not exists(save_dir_run):
                                mkdir(save_dir_run)
                            model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k)
                            model.set_append_matrix(E)
                            print()
                            r_str = f"_rval_{str(r)}"
                            perform_updates(
                                dataset,
                                n_batches,
                                batch_phis,
                                model,
                                method,
                                model.update_svd_bcg,
                                r_str,
                                save_dir_run,
                                res_norms_list,
                                rel_errs_list,
                                make_plots=test["make_plots"],
                                lam_coeff=test["lam_coeff"],
                                r=r,
                            )
                            if test["make_plots"]:
                                fig_dir = save_dir_run + "/figures"
                                if not exists(normpath(fig_dir)):
                                    mkdir(normpath(fig_dir))
                                plot_stacked_relative_errors(
                                    rel_errs_list, batch_phis, fig_dir
                                )
                                plot_stacked_residual_norms(
                                    res_norms_list, batch_phis, fig_dir
                                )
                else:
                    raise ValueError(
                        f"Error: Update method {method} does not exist. Must be one of the following."
                    )
