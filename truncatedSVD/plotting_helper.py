import numpy as np
import matplotlib.pyplot as plt

def plot_residual_norms(errs, filename, title="Residual Norms for Singular Vectors"):
    position = np.arange(errs.shape[0])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    ax.plot(position,errs)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Residual Norm")
    ax.set_yscale('log')
    ax.set_ylim(1e-4,1)
    plt.savefig("../figures/"+filename+".pdf", bbox_inches="tight", pad_inches=0.2)


def plot_relative_errs(errs, filename, title="Relative Error for Singular Vectors"):
    position = np.arange(errs.shape[0])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    ax.plot(position,errs)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Relative Error")
    ax.set_yscale('log')
    ax.set_ylim(1e-8,1)
    plt.savefig("../figures/"+filename+".pdf", bbox_inches="tight", pad_inches=0.2)


def plot_stacked_residual_norms(errs1, errs2, errs3, filename, title="Residual Norms for Singular Vectors"):
    position = np.arange(errs1.shape[0])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    ax.plot(position,errs1,label="$\phi=1$")
    ax.plot(position,errs2,label="$\phi=6$")
    ax.plot(position,errs3,label="$\phi=12$")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Residual Norm")
    ax.set_yscale('log')
    ax.set_ylim(1e-4,1)
    ax.legend()
    plt.savefig("../figures/"+filename+".pdf", bbox_inches="tight", pad_inches=0.2)


def plot_stacked_relative_errs(errs1, errs2, errs3, filename, title="Relative Error for Singular Vectors"):
    position = np.arange(errs1.shape[0])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.grid(True, linewidth=1, linestyle="--", color='k', alpha=0.1)
    ax.tick_params(which="both", direction='in', bottom=True, top=True, left=True, right=True)
    ax.plot(position,errs1,label="$\phi=1$")
    ax.plot(position,errs2,label="$\phi=6$")
    ax.plot(position,errs3,label="$\phi=12$")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_xlim(position[0], position[-1])
    ax.set_ylabel("Relative Error")
    ax.set_yscale('log')
    ax.set_ylim(1e-5,1e-1)
    ax.legend()
    plt.savefig("../figures/"+filename+".pdf", bbox_inches="tight", pad_inches=0.2)
