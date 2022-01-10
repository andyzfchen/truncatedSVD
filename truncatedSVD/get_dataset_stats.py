# from functools import cache
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Script parameters
n_value = 100
# datasets = ["CISI", "CRAN", "MED", "ML1M", "Reuters"]
datasets = ["CISI", "CRAN", "MED"]


# Create cache folder
if not os.path.exists("../cache"):
    os.mkdir("../cache")

Sigma_array_list = []

for dataset in datasets:
    print(f"{dataset} dataset stats.")

    if not os.path.exists("../cache/" + dataset):
        os.mkdir("../cache/" + dataset)

    # Load entire data matrix
    # A_matrix = np.load(f"../datasets/{dataset}/{dataset}.npy")
    
    A_matrix = loadmat(f"../datasets/{dataset}/A_{dataset}.mat")["A_" + f"{dataset}".lower()].toarray()
    
    # Get dimensions
    (n_row, n_col) = np.shape(A_matrix)
    print("Number of rows:    " + str(n_row))
    print("Number of cols:    " + str(n_col))
    print("Number of nnz/row: " + str(np.sum(A_matrix != 0) / n_row))

    # Calculate true SVD (load if already pre-calculated)
    cache_folder = f"../cache/{dataset}/{dataset}"
    if not os.path.exists(f"{cache_folder}_Sigma_array.npy"):
        U_matrix, Sigma_array, VH_matrix = np.linalg.svd(A_matrix)
        np.save(f"{cache_folder}_U_matrix.npy", U_matrix)
        np.save(f"{cache_folder}_Sigma_array.npy", Sigma_array)
        np.save(f"{cache_folder}_VH_matrix.npy", VH_matrix)
    else:
        U_matrix = np.load(f"{cache_folder}_U_matrix.npy")
        Sigma_array = np.load(f"{cache_folder}_Sigma_array.npy")
        VH_matrix = np.load(f"{cache_folder}_VH_matrix.npy")

    Sigma_array_list.append(Sigma_array[:n_value])

    print()

if not os.path.exists("../figures"):
    os.mkdir("../figures")

# Plot singular value profiles
print("Plotting singular value profiles...")
position = np.arange(1, n_value + 1)
fig, ax = plt.subplots(figsize=(4, 3))
ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
ax.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)

for ii, dataset in enumerate(datasets):
    ax.plot(position, Sigma_array_list[ii], label=dataset)

ax.set_title(f"Leading {n_value} Singular Values")
ax.set_xlabel("$i$")
ax.set_xlim(position[0], position[-1])
ax.set_ylabel("$\hat{\sigma}_i$")
ax.set_yscale("log")
ax.set_ylim(1e1, 1e3)
ax.legend(loc="upper right")
plt.savefig(f"../figures/first_{n_value}_singular_values.png", bbox_inches="tight", dpi=200)
plt.close()
