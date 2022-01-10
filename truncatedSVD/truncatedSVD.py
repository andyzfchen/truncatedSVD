#!/usr/bin/env python3

import numpy as np
import EvolvingMatrix as EM
import os
from scipy.io import loadmat


datasets = ["CISI", "CRAN", "MED", "ML1M", "Reuters"]
batch_splits = [10]
phis = [[1, 5, 10]]
update_methods = ["zha-simon", "bcg"]
r_values = [10]
m_percent = 0.10


# debug mode
# datasets = [ "CISI" ]
datasets = ["CISI", "CRAN", "MED"]
batch_splits = [10]
phis = [[1, 5, 10]]
update_methods = ["zha-simon", "bcg"]
r_values = [50]
m_percent = 0.10

# TODO: loop through various values of k (25,50,100)
k_dim = 50

# Create local folder to save outputs
if not os.path.exists("../cache"):
    os.mkdir("../cache")

for dataset in datasets:
    print(f"Using {dataset} dataset.")

    for r_value in r_values:
        print(f"Using r value of {str(r_value)}.")

        for method in update_methods:
            if method == "zha-simon" and r_value != r_values[0]:
                continue

            print(f"Using the {method} evolution method.")

            if not os.path.exists("../cache/" + method):
                os.mkdir("../cache/" + method)

            for n_batches, phi in zip(batch_splits, phis):
                print(
                    f"Performing truncated SVD on dataset {dataset} using batch_split = {str(n_batches)}."
                )

                # Create directory to save data for this batch split
                temp_dir = f"../cache/{method}/{dataset}_batch_split_{str(n_batches)}"
                if not os.path.exists(temp_dir):
                    os.mkdir(temp_dir)

                # Load dataset (.npy)
                # A_full = np.load(f"../datasets/{dataset}/{dataset}.npy").T

                A_full = loadmat(f"../datasets/{dataset}/A_{dataset}.mat")[
                    "A_" + f"{dataset}".lower()
                ]

                # Calculate row index to split data
                (m_dim_full, n_dim) = np.shape(A_full)
                print(f"Data is of shape ({m_dim_full}, {n_dim})")

                m_dim = int(np.ceil(m_dim_full * m_percent))
                s_dim = int(np.floor(m_dim_full * (1 - m_percent)))

                # Split into initial matrix and matrix to be appended
                B = A_full[:m_dim, :]
                E = A_full[m_dim:, :]

                # Initialize EM object with initial matrix and matrix to be appended and set desired rank of truncated SVD
                model = EM.EvolvingMatrix(B, n_batches=n_batches, k_dim=k_dim)
                model.set_append_matrix(E)
                print()

                r_str = ""
                for ii in range(n_batches):
                    print(f"Batch {str(ii+1)}/{str(n_batches)}.")

                    # Evolve matrix by appending new rows
                    model.evolve()

                    # Calculate truncated SVD for updated matrix
                    if method == "zha-simon":
                        model.update_svd_zha_simon()
                    elif method == "bcg":
                        # Uk, Sigmak, VHk = model.evolve_matrix_deflated_bcg(
                        # step_dim=int(np.ceil(s_dim / n_batches)), r_dim=r_value
                        # )
                        model.update_svd_bcg()
                        r_str = "_rval_" + str(r_value)
                    elif method == "brute":
                        model.update_svd_brute_force()
                    elif method == "naive":
                        model.update_svd_naive()
                    elif method == "fd":
                        model.update_fd()
                    else:
                        raise ValueError(
                            f"Error: Update method {method} does not exist. Must be one of the following."
                        )

                    # Save results if batch number specified
                    if model.phi in phi:
                        # Calculate true SVD for this batch
                        model.calculate_true_svd(method, dataset)

                        # Caluclate metrics
                        model.save_metrics(temp_dir, print_metrics=True, r_str=r_str)

                    print()

# def load_experiment_specs(experiment_directory):
#     return None


# def main(experiment_directory, overwrite):

#     # Load experiment specifications
#     specs = load_experiment_specs(experiment_directory)

#     print("Experiment description: " + specs["Description"])
#     print("Overwriting previous data?: " + {specs["Overwrite"]})

#     datasets = specs["Datasets"]
#     print("Running experiments on datasets:")
#     print(*specs["Datasets"], sep="\n")

#     update_methods = specs["UpdateMethods"]
#     print("Running experiments with update methods:")
#     print(*specs["UpdateMethods"], sep="\n")

#     batch_splits = specs["BatchSplits"]
#     print("Running experiments with batch splits:")
#     print(*specs["BatchSplits"], sep="\n")

#     m_percent = specs["m_percent"]
#     print(f"Initial matrix to be formed my leading {m_percent}% of rows.")


# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Run experiments for updating truncated SVD of evolving matrices."
#     )
#     parser.add_argument(
#         "--experiment",
#         "-e",
#         dest="experiment_directory",
#         required=True,
#         help="Experiment directory. This directory includes the experiment specifications in 'specs.json' along with all the datasets."
#         + "All results will be saved in this directory as well.",
#     )
#     parser.add_argument(
#         "--overwrite",
#         dest="overwrite",
#         required=False,
#         default=True,
#         help="Option to overwrite existing results. If True, any existing experimental results will be "
#         + "overwritten with results obtained from the current run.",
#     )

#     args = parser.parse_args()

#     main(args.experiment_directory, args.overwrite)
