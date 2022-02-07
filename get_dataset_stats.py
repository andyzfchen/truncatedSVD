# from functools import cache
import numpy as np
import matplotlib.pyplot as plt
from os import mkdir
from os.path import normpath, exists, join
from scipy.linalg import svdvals
from truncatedSVD.utils import init_figure, check_and_create_dir

# Script parameters
n_sv = 100
datasets = {
    "CISI": "./datasets/CISI/CISI_KKU.npy",
    "CRAN": "./datasets/CRAN/CRAN_KKU.npy",
    "MED": "./datasets/MED/MED_KKU.npy",
    "ML1M": "./datasets/ML1M/ML1M.npy",
    "Reuters": "./datasets/Reuters/Reuters.npy",
}

# Create cache folder
stats_dir = "./cache/dataset_stats"
check_and_create_dir(stats_dir)

sigma_array_list = []

for dataset in datasets.values():
    # Load data
    data = np.load(dataset)

    # Get dimensions
    (n_row, n_col) = np.shape(data)
    print(f"Dataset:           {dataset}")
    print(f"Number of rows:    {str(n_row)}")
    print(f"Number of cols:    {str(n_col)}")
    print(f"Number of nnz/row: {str(np.sum(data != 0) / n_row)}")

    # Calculate singular values of data
    s = svdvals(data)

    # Get first n singular values
    sigma_array_list.append(s[:n_sv])
    print()

# Plot singular value profile for each dataset
print("Plotting singular value profiles.")
fig, ax = init_figure(
    title=f"Leading {n_sv} Singular Values", 
    xlabel="$i$", 
    ylabel="$\hat{\sigma}_i$", 
    fontsize="x-large")

sv_idx = np.arange(1, n_sv + 1)
for name, ss in zip(datasets.keys(), sigma_array_list):
    ax.plot(sv_idx, ss, label=name)

ax.set_xlim(sv_idx[0], sv_idx[-1])
ax.legend(loc="upper right")
plt.savefig(
    normpath(join(stats_dir, f"sv_{n_sv}_profile.png")),
    bbox_inches="tight",
    dpi=200,
)
plt.close()
