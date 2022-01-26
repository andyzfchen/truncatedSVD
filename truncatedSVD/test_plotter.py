import numpy as np
import matplotlib.pyplot as plt
from os.path import join,normpath


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
    
    # if ylim is None:
    #     ax.set_ylim(np.asarray(errs_list).min(), np.asarray(errs_list).max())
    # else:
    #     ax.set_ylim(ylim[0], ylim[1])
    
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
    
    # if ylim is None:
    #     ax.set_ylim(np.asarray(errs_list).min(), np.asarray(errs_list).max())
    # else:
    #     ax.set_ylim(ylim[0], ylim[1])
    
    ax.legend(loc="lower right")
    plt.savefig(normpath(join(save_dir,f"relerrs.png")), bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()

def plot_runtimes(runtimes, xval, filename, title=None):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
    ax.tick_params(which="both", direction="in", bottom=True, top=True, left=True, right=True)
    for x, t in zip(xval, runtimes):
        ax.plot(x, t, label="Update(%i)" % x)

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(xval[0], xval[-1])
    ax.set_ylabel("Runtime (s)")
    ax.set_yscale("log")
    ax.legend(loc="lower right")
    plt.savefig(f"../figures/{filename}.png", bbox_inches="tight", pad_inches=0.2, dpi=200)
    plt.close()