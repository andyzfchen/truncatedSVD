from cProfile import run
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, normpath


def plot_residual_norms(
    errs_list,
    phi_list,
    save_dir,
    title="Residual norms for singular triplets",
    filename="res_norm.png",
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
    title="Relative errors for singular values",
    filename="rel_err.png",
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


def plot_covariance_errors(
    errs_list, phi_list, update_method, title="Covariance Error", filename="cov_err.png"
):
    """Helper function to plot covariance errors.

    Parameters
    ----------
    errs_list : list
        List of covariance errors

    """
    return None

    
def plot_projection_errors(
    errs_list, phi_list, update_method, title="Projection Error", filename="proj_err.png"
):
    """Helper function to plot projection errors
    
    Parameters
    ----------
    errs_lis : list
        List of projection errors    
    """
    return None


def plot_pr_curve(precision_list, recall_list, update_methods, dataset, save_dir, title="Precision-Recall Curve"):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    for p, r, method in zip(precision_list, recall_list, update_methods):
        ax.plot(r, p, label=method)
    
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(loc="upper right")
    plt.savefig(
        normpath(join(save_dir, f"{dataset}_pr_curve.png")), bbox_inches="tight", pad_inches=0.2, dpi=200
    )
    plt.close()


def plot_runtimes(runtimes_dict, filename, title="Runtime"):
    """Helper function to plot runtimes.

    Parameters
    ----------
    runtimes_dict : dict
        Dictionary of runtimes    
    """
    # Plot runtimes for each method
    update_methods = np.unique([tup[0] for tup in list(runtimes_dict.keys())])
    batch_sizes = np.unique([tup[1] for tup in list(runtimes_dict.keys())])    
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    for method in update_methods:
        for n_batches in batch_sizes:
            mkeys = [tup for tup in list(runtimes_dict.keys()) if (method in tup and n_batches in tup)]
            rt_data = np.stack([(mk[2], runtimes_dict[mk]) for mk in mkeys])
            ax.plot(rt_data[:, 0], rt_data[:, 1], label=f"{method}, n_batches={n_batches}")

    ax.set_title(title)
    ax.set_xlabel("$k$")
    ax.set_ylabel("Runtime (s)")
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.savefig(
        normpath(filename), bbox_inches="tight", pad_inches=0.2, dpi=200
    )
    plt.close()
