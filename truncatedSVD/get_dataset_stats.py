import numpy as np
import os
import matplotlib.pyplot as plt

n_value = 100
datasets = ["CISI", "CRAN", "MED", "ML1M", "Reuters"]

if not os.path.exists("../cache"):
    os.mkdir("../cache")

Sigma_array_list = []

for dataset in datasets:
    print(dataset + " dataset stats.")

    if not os.path.exists("../cache/" + dataset):
        os.mkdir("../cache/" + dataset)

    A_matrix = np.load("../datasets/" + dataset + "/" + dataset + ".npy")
    (n_row, n_col) = np.shape(A_matrix)
    print("Number of rows:    " + str(n_row))
    print("Number of cols:    " + str(n_col))
    print("Number of nnz/row: " + str(np.sum(A_matrix != 0) / n_row))

    if not os.path.exists("../cache/" + dataset + "/" + dataset + "_Sigma_array.npy"):
        U_matrix, Sigma_array, VH_matrix = np.linalg.svd(A_matrix)

        np.save("../cache/" + dataset + "/" + dataset + "_U_matrix.npy", U_matrix)
        np.save("../cache/" + dataset + "/" + dataset + "_Sigma_array.npy", Sigma_array)
        np.save("../cache/" + dataset + "/" + dataset + "_VH_matrix.npy", VH_matrix)
    else:
        U_matrix = np.load("../cache/" + dataset + "/" + dataset + "_U_matrix.npy")
        Sigma_array = np.load(
            "../cache/" + dataset + "/" + dataset + "_Sigma_array.npy"
        )
        VH_matrix = np.load("../cache/" + dataset + "/" + dataset + "_VH_matrix.npy")

    Sigma_array_list.append(Sigma_array[:n_value])

    print()

if not os.path.exists("../figures"):
    os.mkdir("../figures")

position = np.arange(1, n_value + 1)

fig, ax = plt.subplots(figsize=(4, 3))
ax.grid(True, linewidth=1, linestyle="--", color="k", alpha=0.1)
ax.tick_params(
    which="both", direction="in", bottom=True, top=True, left=True, right=True
)
for ii, dataset in enumerate(datasets):
    ax.plot(position, Sigma_array_list[ii], label=dataset)
ax.set_title("Leading 100 Singular Values")
ax.set_xlabel("$i$")
ax.set_xlim(position[0], position[-1])
ax.set_ylabel("$\hat{\sigma}_i$")
ax.set_yscale("log")
ax.set_ylim(1e0, 1e5)
ax.legend(loc="upper right")
plt.savefig(
    "../figures/first_100_singular_values.pdf", bbox_inches="tight", pad_inches=0.2
)
plt.close()
