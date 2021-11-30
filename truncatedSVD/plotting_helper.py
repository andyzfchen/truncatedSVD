import numpy as np
import matplotlib.pyplot as plt

def plot_residual_norms(errs_list, evolution_methods, filename, title="Residual Norms for Singular Vectors"):
    position = np.arange(1,errs_list[0].shape[0]+1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    for errs, method in zip(errs_list, evolution_methods):
      ax.plot(position,errs, label=method)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Residual Norm")
    ax.set_yscale('log')
    ax.set_ylim(1e-4,1)
    ax.legend()
    plt.savefig("../figures/"+filename+".pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close()


def plot_relative_errs(errs_list, evolution_methods, filename, title="Relative Error for Singular Vectors"):
    position = np.arange(1,errs_list[0].shape[0]+1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    for errs, method in zip(errs_list, evolution_methods):
      ax.plot(position,errs, label=method)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Relative Error")
    ax.set_yscale('log')
    ax.set_ylim(1e-8,1)
    ax.legend()
    plt.savefig("../figures/"+filename+".pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close()


def plot_stacked_residual_norms(errs_list, phi_list, filename, title="Residual Norms for Singular Vectors"):
    position = np.arange(1,errs_list[0].shape[0]+1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    for errs, phi in zip(errs_list, phi_list):
      ax.plot(position,errs,label="Update(%i)" % phi)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Residual Norm")
    ax.set_yscale('log')
    ax.set_ylim(1e-4,1)
    ax.legend()
    plt.savefig("../figures/"+filename+".pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close()


def plot_stacked_relative_errs(errs_list, phi_list, filename, title="Relative Error for Singular Vectors"):
    position = np.arange(1,errs_list[0].shape[0]+1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    for errs, phi in zip(errs_list, phi_list):
      ax.plot(position,errs,label="Update(%i)" % phi)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Relative Error")
    ax.set_yscale('log')
    ax.set_ylim(1e-8,1)
    ax.legend()
    plt.savefig("../figures/"+filename+".pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close()
