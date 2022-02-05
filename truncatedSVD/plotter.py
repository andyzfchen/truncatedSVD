from cProfile import run
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, normpath


def plot_residual_norms(
    errs_list,
    phi_list,
    save_dir,
    filename="res_norm.png",
    title="Residual norms for singular triplets",   
):
    """Helper function to plot scaled residual norms for singular triplets for each update

    Parameters
    ----------
    errs_list : list
        List of scaled residual norms for each update

    phi_list : list
        List of update indices for each of the arrays in errs_list

    save_dir : str
        Directory in which to save plots

    title : str, default="Residual norms for singular triplets"
        Figure title
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Plot residual norms for each update
    idx = np.arange(1, errs_list[0].shape[0] + 1)
    for errs, phi in zip(errs_list, phi_list):
        ax.plot(idx, errs, label="Update(%i)" % phi)

    # Label figure
    ax.set_title(title)
    ax.set_xlabel("Singular Value Index")
    ax.set_xlim(idx[0], idx[-1])
    ax.set_ylabel("Scaled Residual Norm")
    ax.set_yscale("log")
    ax.legend(loc="lower right")
    plt.savefig(
        normpath(join(save_dir, filename)),
        bbox_inches="tight",
        pad_inches=0.2,
        dpi=200,
    )
    plt.close()


def plot_relative_errors(
    errs_list,
    phi_list,
    save_dir,
    filename="rel_err.png",
    title="Relative errors for singular values"
):
    """Helper function to plot relative errors for singular values for each update

    Parameters
    ----------
    errs_list : list
        List of relative errors for each update

    phi_list : list
        List of update indices for each of the arrays in errs_list

    save_dir : str
        Directory in which to save plots

    title : str, default="Relative errors for singular values"
        Figure title
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Plot relative errors for each update
    idx = np.arange(1, errs_list[0].shape[0] + 1)
    for errs, phi in zip(errs_list, phi_list):
        ax.plot(idx, errs, label="Update(%i)" % phi)

    # Label figure
    ax.set_title(title)
    ax.set_xlabel("Singular Value Index")
    ax.set_xlim(idx[0], idx[-1])
    ax.set_ylabel("Relative Error")
    ax.set_yscale("log")
    ax.legend(loc="lower right")
    plt.savefig(
        normpath(join(save_dir, filename)),
        bbox_inches="tight",
        pad_inches=0.2,
        dpi=200,
    )
    plt.close()
