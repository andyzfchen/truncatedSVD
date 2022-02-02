"""Script to run experiments for updating the truncated SVD of evolving matrices.
"""

from os import mkdir
from os.path import normpath, exists, join
import json
import numpy as np
import truncatedSVD.EvolvingMatrix as EM
from truncatedSVD.plotter import *
from truncatedSVD.utils import check_and_create_dir
from string import Template

# Create string template for image filenames
rel_err_fig_name_fmt = Template(
    '${dataset}_${method}_batch_split_${n_batches}_k_dims_${k_dim}_${r_str}rel_err.png')
res_norm_fig_name_fmt = Template(
    '${dataset}_${method}_batch_split_${n_batches}_k_dims_${k_dim}_${r_str}res_norm.png')


def perform_updates(
    dataset,
    n_batches,
    phi,
    model,
    method,
    update_method,
    r_str,
    save_dir,
    cache_dir,
    res_norms_list,
    rel_errs_list,
    make_plots=False,
    **kwargs
):
    """Perform updates for specified number of batches using update method."""
    for ii in range(n_batches):
        print(f"Batch {str(ii+1)}/{str(n_batches)}.")

        # Evolve matrix by appending new rows
        model.evolve()

        # Calculate truncated SVD of updated matrix
        if not kwargs:
            update_method()
        else:
            update_method(**kwargs)

        # Calculate metrics if batch specified
        if model.phi in phi or ii == n_batches - 1:

            # Calculate true SVD for this batch (load from cache if pre-calculated)
            model.calculate_true_svd(
                normpath(join(cache_dir, dataset, "svd_true")))

            # Caluclate metrics
            if method == "frequent-directions":
                model.save_metrics(
                    save_dir, print_metrics=True, A_idx=model.freq_dir.ell, r_str=r_str
                )
            else:
                model.save_metrics(save_dir, print_metrics=True, r_str=r_str)

            # Save error metrics for plotting
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

        print("")

    # Print relative error and residual norm for last singular triplet after updates
    model.print_metrics(sv_idx=model.k_dim - 1)


def split_data(A, m_percent):
    """Split data row-wise"""
    # Calculate index of split
    m_dim_full = np.shape(A)[0]
    m_dim = int(np.ceil(m_dim_full * m_percent))

    # Split into initial matrix and matrix to be appended
    B = A[:m_dim, :]
    E = A[m_dim:, :]

    return B, E


def get_batch_phis(n_batches, phis_to_plot):
    """Get batch numbers to be saved and plotted."""
    batch_phis = [x for x in phis_to_plot if x <= n_batches]
    if n_batches not in batch_phis:
        batch_phis.append(n_batches)
    return batch_phis


def print_message(dataset, data_shape, method, n_batches, k):
    """Print details for current experiment."""
    print(50 * "*")
    print("")
    print(f"Dataset:           {dataset} {data_shape}")
    print(f"Update method:     {method}")
    print(f"Number of batches: {n_batches}")
    print(f"Rank k of updates: {k}\n")


def make_all_plots(fig_dir):
    """Make all plots per experiment"""
    print("Making plots.")


def run_experiments(specs_json, cache_dir):
    # Open JSON file for experiment specification
    f = open(specs_json)
    try:
        test_spec = json.load(f)
    except ValueError as err:
        print("Tests file is not a valid JSON file. Please double check syntax.")
        exit()

    # Create cache path to save results
    cache_path = join(cache_dir, "cache")
    check_and_create_dir(cache_path)

    # Loop through each dataset
    for test in test_spec["tests"]:
        dataset = test["dataset"]
        check_and_create_dir(join(cache_path, dataset))

        # Load data
        data = np.load(test_spec["dataset_info"][dataset])
        # if data.shape[0] < data.shape[1]:
        # data = data.T

        # Initialize dict for runtimes
        runtimes_dict = dict()

        # Run tests for each update method
        for method in test["methods"]:
            check_and_create_dir(join(cache_path, dataset, method))

            # Loop through number of batches
            for n_batches in test["n_batches"]:

                # Get batch numbers to plot for given batch size
                batch_phis = get_batch_phis(n_batches, test["phis_to_plot"])

                # Calculate data split index
                B, E = split_data(data, test["m_percent"])

                # Loop through desired rank k
                for k in test["k_dims"]:

                    # Print message for current experiment
                    print_message(dataset, data.shape, method, n_batches, k)

                    # Create directory to save data for this batch split and k
                    save_dir = join(
                        cache_path,
                        dataset,
                        method,
                        f"{dataset}_batch_split_{str(n_batches)}_k_dims_{str(k)}",
                    )
                    check_and_create_dir(save_dir)

                    # Initialize list of metrics
                    rel_errs_list = []
                    res_norms_list = []

                    # Update truncated SVD using Frequent Directions
                    if method == "frequent-directions":
                        model = EM.EvolvingMatrix(
                            B, n_batches=n_batches, k_dim=k)
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
                            cache_path,
                            res_norms_list,
                            rel_errs_list,
                            make_plots=test["make_plots"],
                        )

                        if test["make_plots"]:
                            # fig_dir = save_dir + "/figures"
                            fig_dir = join(cache_path, "figures")
                            check_and_create_dir(fig_dir)
                            title = dataset + ", " + \
                                test_spec["method_label"][method]
                            plot_relative_errors(
                                rel_errs_list,
                                batch_phis,
                                fig_dir,
                                filename=rel_err_fig_name_fmt.substitute(
                                    dataset=dataset,
                                    method=method,
                                    n_batches=n_batches,
                                    k_dim=k,
                                    r_str="",
                                ),
                                title=title
                            )
                            plot_residual_norms(
                                res_norms_list,
                                batch_phis,
                                fig_dir,
                                filename=res_norm_fig_name_fmt.substitute(
                                    dataset=dataset,
                                    method=method,
                                    n_batches=n_batches,
                                    k_dim=k,
                                    r_str="",
                                ),
                                title=title
                            )
                    # Update truncated SVD using Zha-Simon projection variation
                    elif method == "zha-simon":
                        model = EM.EvolvingMatrix(
                            B, n_batches=n_batches, k_dim=k)
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
                            cache_path,
                            res_norms_list,
                            rel_errs_list,
                            make_plots=test["make_plots"],
                        )

                        if test["make_plots"]:
                            # fig_dir = save_dir + "/figures"
                            fig_dir = join(cache_path, "figures")
                            check_and_create_dir(fig_dir)
                            title = dataset + ", " + \
                                test_spec["method_label"][method]
                            plot_relative_errors(
                                rel_errs_list,
                                batch_phis,
                                fig_dir,
                                filename=rel_err_fig_name_fmt.substitute(
                                    dataset=dataset,
                                    method=method,
                                    n_batches=n_batches,
                                    k_dim=k,
                                    r_str=""
                                ),
                                title=title
                            )
                            plot_residual_norms(
                                res_norms_list,
                                batch_phis,
                                fig_dir,
                                filename=res_norm_fig_name_fmt.substitute(
                                    dataset=dataset,
                                    method=method,
                                    n_batches=n_batches,
                                    k_dim=k,
                                    r_str=""
                                ),
                                title=title
                            )
                    # Update truncated SVD using enhanced projection variation
                    elif method == "bcg":
                        for r in test["r_values"]:
                            for run_num in range(test["num_runs"]):
                                res_norms_list = []
                                rel_errs_list = []
                                save_dir_run = normpath(
                                    join(save_dir, f"run_{str(run_num)}")
                                )
                                check_and_create_dir(save_dir_run)
                                model = EM.EvolvingMatrix(
                                    B, n_batches=n_batches, k_dim=k)
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
                                    cache_path,
                                    res_norms_list,
                                    rel_errs_list,
                                    make_plots=test["make_plots"],
                                    lam_coeff=test["lam_coeff"],
                                    r=r,
                                )

                                if test["make_plots"]:
                                    # fig_dir = save_dir_run + "/figures"
                                    fig_dir = join(cache_path, "figures")
                                    check_and_create_dir(fig_dir)
                                    title = f"{dataset}, {test_spec['method_label'][method]} (r={r})"
                                    plot_relative_errors(
                                        rel_errs_list,
                                        batch_phis,
                                        fig_dir,
                                        filename=rel_err_fig_name_fmt.substitute(
                                            dataset=dataset,
                                            method=method,
                                            n_batches=n_batches,
                                            k_dim=k,
                                            r_str=f"r_{r}_",
                                        ),
                                        title=title
                                    )
                                    plot_residual_norms(
                                        res_norms_list,
                                        batch_phis,
                                        fig_dir,
                                        filename=res_norm_fig_name_fmt.substitute(
                                            dataset=dataset,
                                            method=method,
                                            n_batches=n_batches,
                                            k_dim=k,
                                            r_str=f"r_{r}_",
                                        ),
                                        title=title
                                    )
                    else:
                        raise ValueError(
                            f"Update method {method} does not exist. Must be one of the following: zha-simon, bcg, brute-force, naive."
                        )

                    # # Make covariance and projection errors plots
                    # if test["make_plots"]:
                    #     fig_dir = join(cache_path, "figures")
                    #     check_and_create_dir(fig_dir)

                    #     plot_covariance_errors(cov_err_list)
                    #     plot_projection_errors(proj_err_list)

                    # Add entry into runtimes dict
                    runtimes_dict[(method, n_batches, k)] = model.runtime

                    print("")

        # Plot runtimes (for each dataset)
        check_and_create_dir(join(cache_path, "figures"))
        rt_plot_filename = join(cache_path, f"figures/{dataset}_runtime.png")
        plot_runtimes(runtimes_dict, rt_plot_filename, f"Runtime ({dataset})")


######################################################################

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Run experiments for updating truncated SVD of evolving matrices.")

    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="specs_json",
        required=True,
        help="Experiment specifications. This file specifies all configurations for the experiments to run."
    )
    arg_parser.add_argument(
        "--cache_dir",
        "-c",
        dest="cache_dir",
        default=".",
        help="Directory to contain cache folder. A folder named 'cache' will be created to save all results."
    )

    args = arg_parser.parse_args()

    run_experiments(args.specs_json, args.cache_dir)
