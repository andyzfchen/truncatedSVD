import numpy as np
import matplotlib.pyplot as plt
from os.path import join,normpath


def plot_residual_norms(errs_list, evolution_methods,save_dir, title="Residual Norms for Singular Vectors", ylim=None):
    position = np.arange(1, errs_list[0].shape[0] + 1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)
    for errs, method in zip(errs_list, evolution_methods):
        ax.plot(position, errs, label=method)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Residual Norm")
    ax.set_yscale("log")

    ax.legend(loc="lower right")
    plt.savefig(normpath(join(save_dir,f"resnorms.png")), bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()


def plot_relative_errors(errs_list, evolution_methods, save_dir, title="Relative Error for Singular Vectors", ylim=None):
    position = np.arange(1, errs_list[0].shape[0] + 1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)
    for errs, method in zip(errs_list, evolution_methods):
        ax.plot(position, errs, label=method)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Relative Error")
    ax.set_yscale("log")
    
    ax.legend(loc="lower right")
    plt.savefig(normpath(join(save_dir,f"relerrs.png")), bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()


def plot_stacked_residual_norms(errs_list, phi_list, save_dir, title="Residual Norms for Singular Vectors", ylim=None):
    position = np.arange(1, errs_list[0].shape[0] + 1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)
    for errs, phi in zip(errs_list, phi_list):
        ax.plot(position, errs, label="Update(%i)" % phi)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Residual Norm")
    ax.set_yscale("log")
    
    ax.legend(loc="lower right")
    plt.savefig(normpath(join(save_dir,f"resnorms.png")), bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()


def plot_stacked_relative_errors(errs_list, phi_list, save_dir, title="Relative Error for Singular Vectors", ylim=None):
    position = np.arange(1, errs_list[0].shape[0] + 1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)
    for errs, phi in zip(errs_list, phi_list):
        ax.plot(position, errs, label="Update(%i)" % phi)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Relative Error")
    ax.set_yscale("log")
    
    ax.legend(loc="lower right")
    plt.savefig(normpath(join(save_dir,f"relerrs.png")), bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()


def plot_runtimes(runtimes, xval, filename, title=None):
    """Helper function to plot runtimes."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)
    for x, t in zip(xval, runtimes):
        ax.plot(x, t, label="Update(%i)" % x)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(xval[0], xval[-1])
    ax.set_ylabel("Runtime (s)")
    
    ax.legend(loc="lower right")
    plt.savefig(f"../figures/{filename}.png", bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()


def plot_errors(cov_err_list, proj_err_list, phi_list, update_methods, filename, title="", ylim=None):
    """Helper function to plot covariance and projection errors."""
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(4, 6), sharex=True)
    ax1.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax2.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax1.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)

    # Plot errors vs. update for each update method
    for method in update_methods:
        for phi, cov_err, proj_err in zip(phi_list, cov_err_list, proj_err_list):
            ax1.plot(phi, cov_err, label="Update(%i)" % phi)
            ax2.plot(phi, proj_err, label="Update(%i)" % phi)
    
    # Figure labels
    ax1.set_title(f"{title} (Covariance Error)")
    ax2.set_title(f"{title} (Projection Error)")
    
    ax1.set_xlabel("Update")
    ax2.set_xlabel("Update") 
    ax1.set_xlim(min(phi_list), max(phi_list))
    
    ax1.set_ylabel("Covariance error")
    ax2.set_ylabel("Projection error")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")
    
    plt.savefig(f"../figures/{filename}.png", bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()
    return None


def plot_pr_curve(precision, recall, leg_name, filename, title=None):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)
    
    for p, r, l in zip(precision, recall, leg_name):
        ax.plot(p, r, label=l) 
    
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(loc="lower left")
    plt.savefig(f"../figures/{filename}.png", bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()
    