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
    ax.set_xlabel("Update Index")
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
    ax.set_xlabel("Update Index")
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


def plot_covariance_projection_errors(
    errs_list,
    phi_list,
    update_methods,
    title="Projection Error",
    filename="proj_err.png",
):
    """Helper function to plot projection errors."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    # Plot errors vs. update for each update method
    for method in update_methods:
        for phi, cov_err, proj_err in zip(phi_list, cov_err_list, proj_err_list):
            ax1.plot(phi, cov_err, label="Update(%i)" % phi)

    # Figure labels
    ax.set_title(f"{title} (Covariance Error)")
    ax.set_xlabel("Update")
    ax.set_xlim(min(phi_list), max(phi_list))
    ax.set_ylabel("Covariance error")
    ax.set_yscale("log")
    ax.legend(loc="lower right")
    plt.savefig(
        f"../figures/{filename}.png", bbox_inches="tight", pad_inches=0.2, dpi=200
    )
    plt.close()
    return None


def plot_pr_curve(precision, recall, leg_name, filename, title=None):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )

    for p, r, l in zip(precision, recall, leg_name):
        ax.plot(p, r, label=l)

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(loc="lower left")
    plt.savefig(
        f"../figures/{filename}.png", bbox_inches="tight", pad_inches=0.2, dpi=200
    )
    plt.close()


def plot_runtimes(runtimes, xval, filename, title="Runtime"):
    """Helper function to plot runtimes."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(
        which="both", direction="in", bottom=True, top=True, left=True, right=True
    )
    for x, t in zip(xval, runtimes):
        ax.plot(x, t, label="Update(%i)" % x)

    ax.set_title(title)
    ax.set_xlabel("Update Index")
    ax.set_xlim(xval[0], xval[-1])
    ax.set_ylabel("Runtime (s)")

    ax.legend(loc="lower right")
    plt.savefig(
        f"../figures/{filename}.png", bbox_inches="tight", pad_inches=0.2, dpi=200
    )
    plt.close()
