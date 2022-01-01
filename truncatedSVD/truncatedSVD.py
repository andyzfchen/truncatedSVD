#!/usr/bin/env python3

import numpy as np
import EvolvingMatrix as EM
import os
from scipy.io import loadmat


datasets = ["CISI", "CRAN", "MED", "ML1M", "Reuters"]
batch_splits = [10]
phis = [[1, 5, 10]]
evolution_methods = ["zha-simon", "bcg"]
r_values = [10]
m_percent = 0.10


# debug mode
# datasets = [ "CISI" ]
datasets = ["CISI", "CRAN", "MED"]
batch_splits = [10]
phis = [[1, 5, 10]]
evolution_methods = ["zha-simon"]
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

        for evolution_method in evolution_methods:
            if evolution_method == "zha-simon" and r_value != r_values[0]:
                continue

            print(f"Using the {evolution_method} evolution method.")

            if not os.path.exists("../cache/" + evolution_method):
                os.mkdir("../cache/" + evolution_method)

            for n_batches, phi in zip(batch_splits, phis):
                print(
                    f"Performing truncated SVD on dataset {dataset} using batch_split = {str(n_batches)}."
                )

                # Create directory to save data for this batch split
                temp_dir = f"../cache/{evolution_method}/{dataset}_batch_split_{str(n_batches)}"
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

                # Update truncated SVD with each batch update
                for ii in range(n_batches):
                    print(f"Batch {str(ii+1)}/{str(n_batches)}.")

                    # Evolve matrix by appending new rows
                    model.evolve()

                    # Calculate truncated SVD for updated matrix
                    if evolution_method == "zha-simon":
                        # Uk, Sigmak, VHk = model.evolve_matrix_zha_simon(step_dim=int(np.ceil(s_dim/batch_split)))
                        model.update_svd_zha_simon()
                        r_str = ""
                    elif evolution_method == "bcg":
                        Uk, Sigmak, VHk = model.evolve_matrix_deflated_bcg(
                            step_dim=int(np.ceil(s_dim / n_batches)), r_dim=r_value
                        )
                        r_str = "_rval_" + str(r_value)
                    elif evolution_method == "brute":
                        # TODO: Update with brute force method
                        pass
                    elif evolution_method == "naive":
                        # TODO: Update with naive method
                        pass
                    else:
                        # TODO: Handle invalid methods
                        pass

                    # Save results if batch number specified
                    if model.phi in phi:
                        # Calculate true SVD for this batch
                        model.calculate_true_svd(evolution_method, dataset)

                        # Caluclate metrics
                        model.save_metrics(temp_dir, print_metrics=True, r_str=r_str)

                    print()


# if __name__ == "__main__":
#   import argparse

#   parser = argparse.ArgumentParser()
#   parser.add_argument("")

#   args = parser.parse_args()
